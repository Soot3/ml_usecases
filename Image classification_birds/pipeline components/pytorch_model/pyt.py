import argparse
def pyt(img_folder):
  import torch
  import torchvision
  import torch.nn as nn
  from tqdm.notebook import tqdm
  import torch.nn.functional as F
  import torchvision.transforms as T
  import torchvision.models as models
  from torch.utils.data import DataLoader
  from torchvision.utils import make_grid
  import joblib


  # store the training images path into a directory
  train_dir = f"{img_folder}/train"
  # store the validation images path into a directory
  val_dir = f"{img_folder}/valid"
  # store test images path into a directory
  test_dir = f"{img_folder}/test"

  # define accuracy function for the model
  def accuracy(out, labels):
      _, preds = torch.max(out, dim=1)
      return torch.tensor(torch.sum(preds == labels).item() / len(preds))

  # function to get the GPU device
  def get_device():
      if torch.cuda.is_available():
          return torch.device("cuda")
      else:
          return torch.device("cpu")

  # function to transfer the data to the GPU device
  def to_device(data, device):
      if isinstance(data, (list, tuple)):
          return [to_device(x, device) for x in data]
      return data.to(device, non_blocking=True)

  # Class instance to load the data from the GPU device
  class DeviceDataLoader():
      def __init__(self, dl, device):
          self.dl = dl
          self.device = device
          
      def __iter__(self):
          for x in self.dl:
              yield to_device(x, self.device)
              
      def __len__(self):
          return len(self.dl)


  # create a class instance of the neural network module and the functions involved  
  class ImageClassificationBase(nn.Module):
      def training_step(self, batch):
          images, labels = batch
          out = self(images)
          loss = F.cross_entropy(out, labels)
          return loss
      
      def validation_step(self, batch):
          images, labels = batch
          out = self(images)
          loss = F.cross_entropy(out, labels)
          acc = accuracy(out, labels)
          return {"val_loss": loss.detach(), "val_acc": acc}
      
      def validation_epoch_end(self, outputs):
          batch_loss = [x["val_loss"] for x in outputs]
          epoch_loss = torch.stack(batch_loss).mean()
          batch_acc = [x["val_acc"] for x in outputs]
          epoch_acc = torch.stack(batch_acc).mean()
          return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}
      
      def epoch_end(self, epoch, epochs, result):
          print("Epoch: [{}/{}], last_lr: {:.6f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
              epoch+1, epochs, result["lrs"][-1], result["train_loss"], result["val_loss"], result["val_acc"]))
          
  # create a class instance of the ResNet18 pretrained model for transfer learning
  class model(ImageClassificationBase):
      def __init__(self, num_classes):
          super().__init__()
          self.network = models.resnet18(pretrained=True)
          number_of_features = self.network.fc.in_features
          self.network.fc = nn.Linear(number_of_features, num_classes)
          
      def forward(self, xb):
          return self.network(xb)
      
      def freeze(self):
          for param in self.network.parameters():
              param.requires_grad= False
          for param in self.network.fc.parameters():
              param.requires_grad= True
          
      def unfreeze(self):
          for param in self.network.parameters():
              param.requires_grad= True    
              
  # disable gradient calculation
  @torch.no_grad()

  # function for model evaluation
  def evaluate(model, val_dl):
      model.eval()
      outputs = [model.validation_step(batch) for batch in val_dl]
      return model.validation_epoch_end(outputs)

  # function to get learning rate optimizer
  def get_lr(optimizer):
      for param_group in optimizer.param_groups:
          return param_group["lr"]

  # function to fit the training set and validation set into the model
  def fit_one_cycle(epochs, max_lr, model, train_dl, val_dl, weight_decay=0, grad_clip=None,
                  opt_func=torch.optim.Adam):
      torch.cuda.empty_cache()
      
      history = []
      
      opt = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
      sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr, epochs=epochs,
                                                steps_per_epoch=len(train_dl))
      
      for epoch in range(epochs):
          model.train()
          train_loss = []
          lrs = []
          for batch in tqdm(train_dl):
              loss = model.training_step(batch)
              train_loss.append(loss)
              loss.backward()
              
              if grad_clip:
                  nn.utils.clip_grad_value_(model.parameters(), grad_clip)
                  
              opt.step()
              opt.zero_grad()
              
              lrs.append(get_lr(opt))
              sched.step()
              
          result = evaluate(model, val_dl)
          result["train_loss"] = torch.stack(train_loss).mean().item()
          result["lrs"] = lrs
          model.epoch_end(epoch, epochs, result)
          history.append(result)
      return history 

  transform_ds = T.Compose([T.Resize((128, 128)),
                      T.RandomHorizontalFlip(),
                      T.ToTensor()
                      ])

  # store the dataset as a subclass of torchvision.datasets
  train_ds = torchvision.datasets.ImageFolder(root=train_dir, transform=transform_ds)
  val_ds = torchvision.datasets.ImageFolder(root=val_dir, transform=transform_ds) 

  # create a batch size for the images
  batch_size = 128

  # Load the dataset from directory in torchvision.datasets
  train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
  val_dl = DataLoader(val_ds, batch_size, num_workers=4, pin_memory=True)

  # display the GPU device
  device = get_device()

  # transfer the training set and validation set to the GPU device data loader
  train_dl = DeviceDataLoader(train_dl, device)
  val_dl = DeviceDataLoader(val_dl, device)


  # ResNet18 model architecture
  model = to_device(model(num_classes=260), device)  
  result = [evaluate(model, val_dl)]
  epochs = 5
  max_lr = 10e-5
  grad_clip = 0.1
  weight_decay = 10e-4
  opt_func = torch.optim.Adam

  history = fit_one_cycle(epochs, max_lr, model, train_dl, val_dl, 
                        weight_decay=weight_decay, 
                        grad_clip=grad_clip,
                        opt_func=opt_func)
  accuracy = [x["val_acc"] for x in history]
  val_loss = [x["val_loss"] for x in history]
  pytorch_metrics = {'loss':val_loss[-1], 'test':accuracy[-1]}
  torch.save(model.state_dict(), "pytorch_model.pt")
  joblib.dump(pytorch_metrics,'pytorch_metrics')
  print(result)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--img_folder')
  args = parser.parse_args()
  pyt(args.img_folder)
