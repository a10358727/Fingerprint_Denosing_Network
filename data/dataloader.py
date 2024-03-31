import glob
import os
from torch.utils.data import Dataset,DataLoader, Subset
from PIL import Image 
import numpy as np


# train data loader
def train_dataloader(opt):
    # get data
    noise_data_path,clear_data_path = load_data_path(opt)
    
    # create datasets 
    # split dataset into train_dataset and val_dataset with ratio 9:1    
    _dataset = TestDataset(noise_data_path, clear_data_path, opt)
    num_img = len(_dataset)  
    print(f'total loading {num_img} images')
    indices = list(range(num_img))
    random_state = np.random.get_state()
    np.random.seed(2023)
    np.random.shuffle(indices)
    np.random.set_state(random_state)
    train_indices, val_indices= (
        indices[:int(num_img * 0.9)],
        indices[int(num_img * 0.9):]
    )
    train_dataset = Subset(_dataset, train_indices)
    val_dataset = Subset(_dataset, val_indices)
    print(f'{len(train_dataset)} images for train')
    print(f'{len(val_dataset)} images for valid')
    
    # create dataloader
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=opt.num_threads, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=opt.num_threads, shuffle=False)
    return train_loader, val_loader

# test data loader 
def test_dataloader(opt):
    # get data
    noise_data_path = load_test_path(opt)
    # create datasets 
    test_dataset = TestDataset(noise_data_path,noise_data_path,opt)
    # create dataloader
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=opt.num_threads, shuffle=False)
    return test_loader

def load_data_path(opt):

    clear_paths = []
    noise_paths = []

    clear_paths = _image_paths_search(opt.clean_dataroot)
    clear_paths.sort()
    noise_paths = _image_paths_search(opt.noise_dataroot)
    noise_paths.sort()

    return clear_paths, noise_paths



def load_test_path(opt):

    noise_paths = []


    noise_paths = _image_paths_search(opt.noise_dataroot)
    noise_paths.sort()


    return noise_paths

def _image_paths_search(image_dirs):
        file_patterns = ['*.bmp', '*.png', '*.jpg', '*.jpeg']
        image_list = []
        for image_dir in image_dirs:
            temp_list = [file for pattern in file_patterns for file in glob.glob(os.path.join(image_dir, '**', pattern), recursive=True)]
            image_list.extend(temp_list)

        print(len(image_list))
        return image_list




class TestDataset(Dataset):
    def __init__(self, noise, clean ,opt):
        super().__init__()
        self.noise_paths = noise
        self.clean_paths = clean
        
        self.input_h = opt.input_h
        self.input_w = opt.input_w
    
    def __len__(self):
        return len(self.noise_paths)
    
    def __getitem__(self, index):
        
        noise_path = self.noise_paths[index]
        clean_path = self.clean_paths[index]
        
        image = Image.open(noise_path).convert("L")
        image = np.array(image)
        # 裁剪圖像
        # image = image[3:-3, 3:-3]
        # # 重新調整形狀並進行正規化
        image = image.reshape([1, self.input_h , self.input_w ]) / 255.0
        
        label = Image.open(clean_path).convert("L")
        label = np.array(label)
        # label = label[3:-3, 3:-3]
        label = label.reshape([1,self.input_h  ,self.input_w ]) / 255.0
        
        level = noise_path.split('/')[-2]   # get the level 
        level = int(level)
        return image, label, level
    
    
# if __name__=='__main__':
#     TestDataset