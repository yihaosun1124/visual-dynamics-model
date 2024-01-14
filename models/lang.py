import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, AutoConfig


class LangEncoder(nn.Module):
  """Language Encoder Module (Distilbert)"""
  def __init__(self, nq = None, finetune = False, aug=False, scratch=False, device='cuda'):
    super().__init__()
    self.finetune = finetune
    self.scratch = scratch # train from scratch vs load weights
    self.aug = aug
    self.device = device
    self.tokenizer = AutoTokenizer.from_pretrained("/home/yihaosun/code/bert")
    if not self.scratch:
      self.model = AutoModel.from_pretrained("/home/yihaosun/code/bert").to(device)
    else:
      self.model = AutoModel.from_config(config = AutoConfig.from_pretrained("distilbert-base-uncased")).to(device)
    self.lang_size = 768
      
  def forward(self, langs):
    try:
      langs = langs.tolist()
    except:
      pass
    
    if self.finetune:
      encoded_input = self.tokenizer(langs, return_tensors='pt', padding=True)
      input_ids = encoded_input['input_ids'].to(self.device)
      attention_mask = encoded_input['attention_mask'].to(self.device)
      lang_embedding = self.model(input_ids, attention_mask=attention_mask)[0][:, -1]
    else:
      with torch.no_grad():
        encoded_input = self.tokenizer(langs, return_tensors='pt', padding=True)
        input_ids = encoded_input['input_ids'].to(self.device)
        attention_mask = encoded_input['attention_mask'].to(self.device)
        lang_embedding = self.model(input_ids, attention_mask=attention_mask)[0][:, -1]
    if self.aug:
      lang_embedding +=  torch.distributions.Uniform(-0.1, 0.1).sample(lang_embedding.shape).cuda()
    return lang_embedding