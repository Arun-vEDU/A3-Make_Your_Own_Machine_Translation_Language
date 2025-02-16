import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# File paths
model_path = "best-model-additive-attentionTransformer.pt"
source_vocab_path = "vocabAA_en.pt"  # English vocabulary
target_vocab_path = "vocabAA_si.pt"  # Sinhala vocabulary

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load vocabularies
try:
    source_vocab = torch.load(source_vocab_path)
    target_vocab = torch.load(target_vocab_path)
except Exception as e:
    print(f"Error loading vocab files: {e}")
    exit()

# Define source and target languages
SRC_LANGUAGE = "en"  # English
TRG_LANGUAGE = "si"  # Sinhala

# Define special tokens
SOS_IDX = target_vocab['<sos>']  # Start-of-sequence token index
EOS_IDX = target_vocab['<eos>']  # End-of-sequence token index

# Define text_transform (tokenization functions)
text_transform = {
    SRC_LANGUAGE: lambda sentence: torch.tensor(
        [source_vocab[token] if token in source_vocab else source_vocab['<unk>'] for token in sentence.lower().split()],
        dtype=torch.long
    ),
    TRG_LANGUAGE: lambda sentence: torch.tensor(
        [target_vocab[token] if token in target_vocab else target_vocab['<unk>'] for token in sentence.lower().split()],
        dtype=torch.long
    )
}

# Define vocab_transform (vocabulary objects)
vocab_transform = {
    SRC_LANGUAGE: source_vocab,
    TRG_LANGUAGE: target_vocab
}


# Define model hyperparameters
input_dim = len(source_vocab)
output_dim = len(target_vocab)
hid_dim = 256
enc_layers = 2
dec_layers = 2
enc_heads = 8
dec_heads = 8
enc_pf_dim = 512
dec_pf_dim = 512
enc_dropout = 0.1
dec_dropout = 0.1
PAD_IDX = source_vocab['<pad>']  # Assuming <pad> exists in both vocabs

# Model architecture
# Design the model
class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = AdditiveAttention(hid_dim, n_heads, dropout, device)
        self.feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        # Self-attention
        _src, _ = self.self_attention(src, src, src, src_mask)
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        # Feedforward
        _src = self.feedforward(src)
        src = self.ff_layer_norm(src + self.dropout(_src))
        return src

#Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length = 329):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers        = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
                                           for _ in range(n_layers)])
        self.dropout       = nn.Dropout(dropout)
        self.scale         = torch.sqrt(torch.FloatTensor([hid_dim])).to(self.device)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len]
        #src_mask = [batch size, 1, 1, src len]
        
        batch_size = src.shape[0]
        src_len    = src.shape[1]
        
        pos        = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        #pos: [batch_size, src_len]
        
        src        = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        #src: [batch_size, src_len, hid_dim]
        
        for layer in self.layers:
            src = layer(src, src_mask)
        #src: [batch_size, src_len, hid_dim]
        
        return src
    
# AdditiveAttention Class (Updated)
class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads=None, dropout=None, device=None):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        self.device = device if device is not None else torch.device("cpu")

    def forward(self, query, key, value, mask=None):
        """
        query: [batch_size, trg_len, hid_dim]
        key: [batch_size, src_len, hid_dim]
        value: [batch_size, src_len, hid_dim]
        mask: [batch_size, 1, 1, src_len]
        """
        batch_size, trg_len, hid_dim = query.size()
        src_len = key.size(1)

        # Expand query and key for additive attention
        query_expanded = query.unsqueeze(2).expand(-1, -1, src_len, -1)  # [B, trg_len, src_len, H]
        key_expanded = key.unsqueeze(1).expand(-1, trg_len, -1, -1)       # [B, trg_len, src_len, H]

        # Calculate energy scores
        energy = torch.tanh(self.W1(key_expanded) + self.W2(query_expanded))  # [B, trg_len, src_len, H]
        attention_scores = self.V(energy).squeeze(-1)  # [B, trg_len, src_len]

        # Apply mask (if provided)
        if mask is not None:
            # Convert mask from [B, 1, 1, src_len] to [B, 1, src_len]
            mask = mask.squeeze(1)  # Remove redundant dimensions
            # Expand mask to match attention_scores dimensions
            mask = mask.expand(-1, trg_len, -1)  # [B, trg_len, src_len]
            attention_scores = attention_scores.masked_fill(mask == 0, -1e10)

        # Compute attention weights
        attention = torch.softmax(attention_scores, dim=-1)  # [B, trg_len, src_len]
        attention = self.dropout(attention)

        # Calculate context vector
        context = torch.bmm(attention, value)  # [B, trg_len, H]

        return context, attention


    
# PositionwiseFeedforwardLayer    
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(hid_dim, pf_dim)
        self.fc2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        #x = [batch size, src len, hid dim]
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x   
    
#decoder

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = AdditiveAttention(hid_dim, None, dropout, device)
        self.encoder_attention = AdditiveAttention(hid_dim, None, dropout, device)
        self.feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        # Self-attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        
        # Encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        
        # Feedforward
        _trg = self.feedforward(trg)
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        return trg, attention

class Decoder(nn.Module):
    
    def __init__(self, output_dim, hid_dim, n_layers, pf_dim, dropout, device, max_length=329):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([
            DecoderLayer(hid_dim, pf_dim, dropout, device) 
            for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
            
        output = self.fc_out(trg)
        return output, attention
    
class Seq2SeqTransformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        
        #src = [batch size, src len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def make_trg_mask(self, trg):
        
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        #trg_pad_mask = [batch size, 1, 1, trg len]
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

    def forward(self, src, trg):
        
        #src = [batch size, src len]
        #trg = [batch size, trg len]
                
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        
        enc_src = self.encoder(src, src_mask)
        #enc_src = [batch size, src len, hid dim]
                
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return output, attention

# Traning    
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
        


enc = Encoder(input_dim, 
              hid_dim, 
              enc_layers, 
              enc_heads, 
              enc_pf_dim, 
              enc_dropout, 
              device, max_length=329
) #max_length

dec = Decoder(
                output_dim=output_dim,
                hid_dim=hid_dim,
                n_layers=dec_layers,
                pf_dim=dec_pf_dim,
                dropout=dec_dropout,
                device=device,
                max_length=329 #max_length  # Passed only once
)

model = Seq2SeqTransformer(enc, dec, PAD_IDX, PAD_IDX, device).to(device)
model.apply(initialize_weights)
# Load the state_dict
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Translate function
def translate_sentence(model, sentence, text_transform, SRC_LANGUAGE, TRG_LANGUAGE, vocab_transform, device, max_length=50):
    model.eval()
    
    # Preprocess input sentence
    src_tensor = text_transform[SRC_LANGUAGE](sentence).unsqueeze(0).to(device)  # Shape: [1, seq_len]
    
    # Create a tensor for the generated output, starting with <sos>
    trg_indexes = [SOS_IDX]
    
    with torch.no_grad():
        for _ in range(max_length):
            trg_tensor = torch.tensor(trg_indexes).unsqueeze(0).to(device)  # Shape: [1, current_seq_len]
            
            # Forward pass through the model (only src and trg)
            output, _ = model(src_tensor, trg_tensor)
            
            # Get the predicted token (last token in sequence)
            pred_token = output.argmax(2)[:, -1].item()
            
            # Stop if <eos> token is generated
            if pred_token == EOS_IDX:
                break  # Break before appending <eos>
            
            # Append to the output sequence
            trg_indexes.append(pred_token)

    # Convert token indexes to words
    trg_tokens = [vocab_transform[TRG_LANGUAGE].get_itos()[idx] for idx in trg_indexes[1:]]  # Skip <sos>
    return " ".join(trg_tokens)
    # Convert token indexes to words
    trg_tokens = [vocab_transform[TRG_LANGUAGE].get_itos()[idx] for idx in trg_indexes[1:]]  # Skip <sos>
    return " ".join(trg_tokens)

# Create Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([
    html.H1("English to Sinhala Translator", className="text-center mb-4"),
    
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H4("English Input:", className="mb-3"),
                dcc.Textarea(
                    id='input-text',
                    placeholder='Enter English text...',
                    style={'width': '100%', 'height': 100},
                    className='mb-3'
                ),
                dbc.Button(
                    "Translate",
                    id='translate-btn',
                    color='primary',
                    className='mb-4'
                ),
            ], md=6),
            
            dbc.Col([
                html.H4("Sinhala Translation:", className="mb-3"),
                html.Div(
                    id='output-text',
                    style={
                        'minHeight': '100px',
                        'border': '1px solid #dee2e6',
                        'borderRadius': '4px',
                        'padding': '10px',
                        'fontSize': '1.2em'
                    },
                    className='bg-light p-3'
                )
            ], md=6)
        ])
    ], fluid=True)
])

@app.callback(
    Output('output-text', 'children'),
    [Input('translate-btn', 'n_clicks')],
    [State('input-text', 'value')]
)
def update_output(n_clicks, input_text):
    if not input_text:
        return "Enter text and click Translate"
    
    try:
        # Translate the input sentence
        translated = translate_sentence(
            model=model,
            sentence=input_text,
            text_transform=text_transform,
            SRC_LANGUAGE=SRC_LANGUAGE,
            TRG_LANGUAGE=TRG_LANGUAGE,
            vocab_transform=vocab_transform,
            device=device
        )
        return translated
    except Exception as e:
        return f"Translation error: {str(e)}"

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)