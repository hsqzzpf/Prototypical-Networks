
import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
import os
import requests

ROOT_PATH = 'miniImageNet/'
image_path = "images.zip"

class MiniImageNet(Dataset):

    

    def __init__(self, setname, download = True):
        self.url = "https://drive.google.com/uc?export=download&id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE"
        id_index = self.url.index('id=') + 2
        self.id = self.url[id_index:]
        self.file_path = os.path.join(ROOT_PATH, image_path)

        if download:
            self.download_file_from_google_drive()

        csv_path = osp.join(ROOT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(ROOT_PATH, 'images', name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label
        self.y = label

        self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

    def _check_exists(self):
        return os.path.exists(os.path.join(ROOT_PATH, 'images/'))

    def download_file_from_google_drive(self):
        import zipfile

        if self._check_exists():
            return

        print("Start Downloading Dataset From Google Drive...(The dataset is 3GB so please wait for a while)")
        def get_confirm_token(response):
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    return value

            return None

        def save_response_content(response):
            CHUNK_SIZE = 32768

            with open(self.file_path, "wb") as f:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)

        

        session = requests.Session()

        response = session.get(self.url, params = { 'id' : self.id }, stream = True)
        token = get_confirm_token(response)

        if token:
            params = { 'id' : self.id, 'confirm' : token }
            response = session.get(self.url, params = params, stream = True)

        save_response_content(response)  
        print("Finish Download Dataset")

        print("Start Unzipping images.zip...")
        zip_ref = zipfile.ZipFile(self.file_path, 'r')
        zip_ref.extractall(ROOT_PATH)
        zip_ref.close()
        print("Finish Unzip")

    