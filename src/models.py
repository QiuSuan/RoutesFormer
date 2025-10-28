"""
RoutesFormer Transformer Model Module

Contains Transformer-based sequence-to-sequence path inference model
"""
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import math
import numpy as np
import logging

logger = logging.getLogger(__name__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    """
    Positional encoding module
    
    Add positional information to each position in sequence
    """
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Initialize positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        
        # Apply sin and cos functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer("pe", pe)
    
    def forward(self, x):
        """
        Args:
            x: Embedded input, shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Transformer model core
    
    Based on encoder-decoder architecture，For sequence-to-sequence path inference
    """
    def __init__(
        self,
        token_indexs: dict,
        is_onehot_embedding: bool,
        use_attributes_tgt: bool,
        src_input_size: tuple,
        tgt_input_size: tuple,
        embedding_size: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 512
    ):
        super(TransformerModel, self).__init__()
        self.token_indexs = token_indexs
        self.is_onehot_embedding = is_onehot_embedding
        self.use_attributes_tgt = use_attributes_tgt
        self.verbose = False
        
        # Source sequence embedding layer
        if self.is_onehot_embedding:
            self.src_input_embedding = nn.Linear(src_input_size[2], embedding_size)
        else:
            self.src_input_embedding1 = nn.Embedding(
                self.token_indexs['pos'] + 1,
                int(embedding_size / 2)
            )
            self.src_input_embedding2 = nn.Linear(
                src_input_size[2] - 1,
                embedding_size - int(embedding_size / 2)
            )
        
        # Target sequenceembedding layer
        if use_attributes_tgt:
            if self.is_onehot_embedding:
                self.tgt_input_embedding = nn.Linear(tgt_input_size[2], embedding_size)
            else:
                self.tgt_input_embedding1 = nn.Embedding(
                    self.token_indexs['pos'] + 1,
                    int(embedding_size / 2)
                )
                self.tgt_input_embedding2 = nn.Linear(
                    tgt_input_size[2] - 1,
                    embedding_size - int(embedding_size / 2)
                )
        else:
            self.tgt_input_embedding = nn.Embedding(
                self.token_indexs['pos'] + 1,
                embedding_size
            )
        
        # Transformer core
        self.transformer = nn.Transformer(
            d_model=embedding_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(embedding_size, dropout=0)
        
        # prediction layer
        self.predictor = nn.Linear(embedding_size, self.token_indexs['pos'] + 1)
        self.bn = nn.BatchNorm1d(embedding_size)
    
    @staticmethod
    def get_key_padding_mask(tokens, type: str, token_indexs: dict, is_onehot_embedding: bool):
        """Generate key padding mask"""
        if type == 'src':
            if is_onehot_embedding:
                key_padding_mask = torch.zeros(tokens.size()[:2], dtype=torch.bool)
                key_padding_mask[tokens[:, :, token_indexs['pos']] == 1] = True
            else:
                key_padding_mask = torch.zeros(tokens.size()[:2], dtype=torch.bool)
                key_padding_mask[tokens[:, :, 0].long() == token_indexs['pos']] = True
        elif type == 'tgt':
            key_padding_mask = torch.zeros(tokens.size(), dtype=torch.bool)
            key_padding_mask[tokens == token_indexs['pos']] = True
        return key_padding_mask
    
    def forward(
        self,
        src,
        tgt,
        decoder_masked: bool,
        is_positional_encoding: bool
    ):
        """
        Forward propagation
        
        Args:
            src: Source sequence (discontinuous path)
            tgt: Target sequence (generated path)
            decoder_masked: Whether to apply mask to decoder
            is_positional_encoding: Whether to use positional encoding
        """
        src_input_size = src.shape
        
        # Generate target sequence mask (Prevent seeing future information)
        if torch.__version__[:3] == '1.7':
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                self, sz=tgt.size()[1]
            ).to(device)
        else:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                sz=tgt.size()[1]
            ).to(device)
        
        
        # Adjust decoder mask (Optional)
        if decoder_masked:
            sz = tgt.size()[1]
            mask = (torch.triu(torch.ones(sz, sz), diagonal=1 - decoder_masked) == 1)
            mask = mask.float().masked_fill(mask == 0, -1e10).masked_fill(mask == 1, float(0.0)).to(device)
            tgt_mask += mask
        
        
        # 生成padding掩码
        src_key_padding_mask = self.get_key_padding_mask(
            src, 'src', self.token_indexs, self.is_onehot_embedding
        ).to(device)
        
        if self.use_attributes_tgt:
            tgt_key_padding_mask = self.get_key_padding_mask(
                tgt, 'src', self.token_indexs, self.is_onehot_embedding
            ).to(device)
        else:
            tgt_key_padding_mask = self.get_key_padding_mask(
                tgt, 'tgt', self.token_indexs, self.is_onehot_embedding
            ).to(device)
        
        # 源序列嵌入
        if self.is_onehot_embedding:
            src = self.src_input_embedding(src)
        else:
            src1 = self.src_input_embedding1(src[:, :, 0].long())
            src2 = self.src_input_embedding2(src[:, :, 1:])
            src = torch.cat((src1, src2), dim=-1)
        
        # Batch Normalization
        src = src.reshape(-1, src.shape[-1])
        src = self.bn(src)
        src = src.reshape(-1, src_input_size[1], src.shape[-1])
        
        # Target sequence嵌入
        if self.use_attributes_tgt:
            if self.is_onehot_embedding:
                tgt = self.tgt_input_embedding(tgt)
            else:
                tgt1 = self.tgt_input_embedding1(tgt[:, :, 0].long())
                tgt2 = self.tgt_input_embedding2(tgt[:, :, 1:])
                tgt = torch.cat((tgt1, tgt2), dim=-1)
        else:
            tgt = self.tgt_input_embedding(tgt)
        
        # Positional encoding
        if is_positional_encoding:
            src = self.positional_encoding(src)
            tgt = self.positional_encoding(tgt)
        
        # Adjust dimension order (PyTorch Transformer requires)
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)
        
        # TransformerForward propagation
        out = self.transformer(
            src, tgt,
            src_mask=None,
            tgt_mask=tgt_mask,
            memory_mask=None,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        return out


class RoutesFormerTransformer:
    """
    RoutesFormer Transformer model wrapper class
    
    Provides training and prediction interfaces
    """
    def __init__(self, model_config, attributes_dict: dict, token_indexs: dict):
        """
        Initialize model
        
        Args:
            model_config: Model config object
            attributes_dict: Attribute dictionary
            token_indexs: Special token index
        """
        self.model_name = 'RoutesFormerTransformer'
        
        # Configuration parameters
        self.is_onehot_embedding = (
            'onehot_embedding' not in attributes_dict
        )
        self.use_attributes_tgt = False
        self.embedding_size = model_config.embedding_size
        self.nhead = model_config.nhead
        self.num_encoder_layers = model_config.num_encoder_layers
        self.num_decoder_layers = model_config.num_decoder_layers
        self.dim_feedforward = model_config.dim_feedforward
        self.lr = model_config.learning_rate
        self.batch_size = model_config.batch_size
        self.epoch_num = model_config.epoch_num
        
        # Mask settings
        self.train_positional_encoding = model_config.train_positional_encoding
        self.eval_positional_encoding = model_config.eval_positional_encoding
        self.train_decoder_masked = model_config.train_decoder_masked
        self.eval_decoder_masked = model_config.eval_decoder_masked
        
        self.attributes_dict = attributes_dict
        self.token_indexs = token_indexs
        self.model = None
    
    def train(self, train_srcs, train_tgts, train_tgts_y, continue_training=False, start_epoch=0):
        """
        Train model
        
        Args:
            train_srcs: training source sequence
            train_tgts: training target sequence（input）
            train_tgts_y: training target sequence（label）
            continue_training: Whether to continue training（not re-initialize model）
            start_epoch: Start epoch number（for logging）
        """
        logger.info(f"Training data shape: src={train_srcs.shape}, tgt={train_tgts.shape}, tgt_y={train_tgts_y.shape}")
        
        # Convert to Tensor
        train_srcs = torch.from_numpy(train_srcs).float().to(device)
        if self.use_attributes_tgt:
            train_tgts = torch.from_numpy(train_tgts).float().to(device)
        else:
            train_tgts = torch.from_numpy(train_tgts).long().to(device)
        train_tgts_y = torch.from_numpy(train_tgts_y).long().to(device)
        
        # Loss function（Fixed to use classification task）
        criteria = nn.CrossEntropyLoss()
        
        # Create data loader
        dataset = TensorDataset(train_srcs, train_tgts, train_tgts_y)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model（only when training for the first time）
        if not continue_training or self.model is None:
            logger.info("Initialize new model...")
            self.model = TransformerModel(
                self.token_indexs,
                self.is_onehot_embedding,
                self.use_attributes_tgt,
                train_srcs.shape,
                train_tgts.shape,
                self.embedding_size,
                self.nhead,
                self.num_encoder_layers,
                self.num_decoder_layers,
                self.dim_feedforward
            ).to(device)
        else:
            logger.info("Continue training existing model...")
        
        self.model.train()
        
        # Initialize or reuse optimizer
        if not continue_training or not hasattr(self, 'optimizer'):
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Training loop
        if start_epoch == 0:
            logger.info(f"Start training, total {self.epoch_num} epochs")
        else:
            logger.info(f"Continue training, epoch {start_epoch + 1} - {start_epoch + self.epoch_num}")
        
        total_loss = 0
        for epoch in range(self.epoch_num):
            for _, data in enumerate(train_loader):
                src, tgt, tgt_y = data
                n_tokens = (tgt_y != self.token_indexs['pos']).sum()
                
                self.optimizer.zero_grad()
                
                # Forward propagation
                out = self.model(
                    src, tgt,
                    False,  # decoder_masked
                    self.train_positional_encoding
                )
                
                # 预测
                out = self.model.predictor(out).permute(1, 0, 2)
                
                # Calculate loss（classification task）
                loss = criteria(
                    out.contiguous().view(-1, out.size(-1)),
                    tgt_y.contiguous().view(-1)
                ) / n_tokens
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss
            
            # Periodically output logs
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / 20
                actual_epoch = start_epoch + epoch + 1
                logger.info(f"epoch {actual_epoch}, average loss: {avg_loss:.6f}")
                print(f"epoch {actual_epoch}, total_loss: {avg_loss:.6f}")
                total_loss = 0
        
        logger.info("Training completed")
    
    def predict(self, srcs, tgts):
        """
        Predict next link
        
        Args:
            srcs: Source sequence
            tgts: Generated target sequence
        
        Returns:
            Predicted probability distribution
        """
        if len(srcs.shape) < 3:
            srcs = srcs.unsqueeze(0)
        srcs = srcs.to(device)
        
        if len(tgts.shape) < 2:
            tgts = tgts.unsqueeze(0)
        tgts = tgts.to(device)
        
        self.model.eval()
        with torch.no_grad():
            out = self.model(
                srcs, tgts,
                False,  # decoder_masked
                self.eval_positional_encoding
            )
            predict_vals = self.model.predictor(out)
            predict_vals = nn.functional.softmax(predict_vals, 2)
        
        return predict_vals