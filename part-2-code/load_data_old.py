import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):
    def __init__(self, data_folder, split):
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.encoder_inputs, self.decoder_targets = self.process_data(data_folder, split)
        
    def process_data(self, data_folder, split):
        # Build file paths
        nl_path = os.path.join(data_folder, f'{split}.nl')
        
        # Check the NL file exists
        if not os.path.exists(nl_path):
            raise FileNotFoundError(f"Natural language file not found: {nl_path}")
        
        nl_lines = self.load_lines(nl_path)
        
        encoder_inputs = []
        for line in nl_lines:
            # Add the task prefix
            # processed_text = f"translate English to SQL: {line.strip()}"
            processed_text = f"{line.strip()}"
            encoder_inputs.append(processed_text)
        
        # The test split has no SQL labels
        if split == "test":
            return encoder_inputs, None
        
        # For train and dev splits load SQL files
        sql_path = os.path.join(data_folder, f'{split}.sql')
        if not os.path.exists(sql_path):
            raise FileNotFoundError(f"SQL file not found: {sql_path}")
            
        sql_lines = self.load_lines(sql_path)
        
        # Validate data alignment
        if len(nl_lines) != len(sql_lines):
            print(f"Warning: NL lines ({len(nl_lines)}) and SQL lines ({len(sql_lines)}) count mismatch for {split}")
        
        decoder_targets = []
        for sql in sql_lines:
            decoder_targets.append(sql.strip())
            
        return encoder_inputs, decoder_targets
    
    def load_lines(self, path):
        """Read a file and return stripped lines"""
        with open(path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        return lines

    def __len__(self):
        return len(self.encoder_inputs)

    def __getitem__(self, idx):
        encoder_text = self.encoder_inputs[idx]
        
        # Tokenize encoder input
        encoder_tokens = self.tokenizer(
            encoder_text, 
            padding=False, 
            truncation=True, 
            max_length=512,
            return_tensors=None
        )
        encoder_ids = encoder_tokens['input_ids']
        
        if self.split == "test":
            return encoder_ids, None, None
        
        # Tokenize decoder target
        decoder_text = self.decoder_targets[idx]
        decoder_tokens = self.tokenizer(
            decoder_text,
            padding=False,
            truncation=True,
            max_length=512,
            return_tensors=None
        )
        decoder_ids = decoder_tokens['input_ids']
        
        # Create decoder input (shifted right)
        decoder_input_ids = [self.tokenizer.pad_token_id] + decoder_ids[:-1]
        
        return encoder_ids, decoder_input_ids, decoder_ids

def normal_collate_fn(batch):
    """Collate function for train/dev splits"""
    encoder_ids_list = []
    decoder_inputs_list = []
    decoder_targets_list = []
    
    for encoder_ids, decoder_input_ids, decoder_target_ids in batch:
        encoder_ids_list.append(torch.tensor(encoder_ids, dtype=torch.long))
        if decoder_input_ids is not None:
            decoder_inputs_list.append(torch.tensor(decoder_input_ids, dtype=torch.long))
            decoder_targets_list.append(torch.tensor(decoder_target_ids, dtype=torch.long))
    
    # Pad sequences
    encoder_ids = pad_sequence(encoder_ids_list, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).float()
    
    if decoder_inputs_list:
        decoder_inputs = pad_sequence(decoder_inputs_list, batch_first=True, padding_value=PAD_IDX)
        decoder_targets = pad_sequence(decoder_targets_list, batch_first=True, padding_value=PAD_IDX)
        initial_decoder_inputs = decoder_inputs[:, 0].unsqueeze(1)
    else:
        decoder_inputs = torch.tensor([], dtype=torch.long)
        decoder_targets = torch.tensor([], dtype=torch.long)
        initial_decoder_inputs = torch.tensor([], dtype=torch.long)
    
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    """Collate function for the test split"""
    encoder_ids_list = []
    
    for encoder_ids, _, _ in batch:
        encoder_ids_list.append(torch.tensor(encoder_ids, dtype=torch.long))
    
    encoder_ids = pad_sequence(encoder_ids_list, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).float()
    
    # Test batches start decoder input with pad_token_id
    batch_size = len(batch)
    initial_decoder_inputs = torch.full((batch_size, 1), PAD_IDX, dtype=torch.long)
    
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    """Create a data loader"""
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = (split == "train")
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(
        dset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_fn, 
        num_workers=0
    )
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    """Load all datasets and build dataloaders"""
    # Ensure the data directory exists
    if not os.path.exists('data'):
        raise FileNotFoundError("Data directory 'data/' not found")
    
    # Verify required files are present
    required_files = ['train.nl', 'train.sql', 'dev.nl', 'dev.sql', 'test.nl']
    for file in required_files:
        if not os.path.exists(os.path.join('data', file)):
            raise FileNotFoundError(f"Required file not found: data/{file}")
    
    print("Loading training data...")
    train_loader = get_dataloader(batch_size, "train")
    
    print("Loading development data...")
    dev_loader = get_dataloader(test_batch_size, "dev")
    
    print("Loading test data...")
    test_loader = get_dataloader(test_batch_size, "test")
    
    print(f"Data loaded: train={len(train_loader.dataset)}, dev={len(dev_loader.dataset)}, test={len(test_loader.dataset)}")
    
    return train_loader, dev_loader, test_loader
