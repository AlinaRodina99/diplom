import telebot
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image, ImageFile
import numpy as np
import torch
from torch import nn
import torch.optim as optim


bot = telebot.TeleBot('token')

@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, 'Hi! You can send me a painting and I will tell you who the artist is! ')

@bot.message_handler(content_types=['photo', 'document'])
def handle_docs_photo(message):
    if message.content_type == 'photo':
        file_info = bot.get_file(message.photo[len(message.photo)-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        src = '/home/alina/Downloads/file_0.jpg'
        with open(src, 'wb') as new_file:
           new_file.write(downloaded_file)
        bot.send_message(message.chat.id, 'Painting has been received!')

        artist = identifyArtistByPhoto(src)
        bot.send_message(message.chat.id, f'Painter: {artist}')
    elif message.content_type == 'document':
        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        src = '/home/alina/Downloads/file_0.jpg'
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)
        bot.reply_to(message, "Painting has been received!")

        artist = identifyArtistByPhoto(src)
        bot.send_message(message.chat.id, f'Painter: {artist}')

t1 = transforms.ToTensor()
t2 = transforms.Resize(224)
t3 = transforms.CenterCrop(224)
t4 = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

ts = [t1, t2, t3, t4]

df = pd.read_csv("new_data_labels(2).csv", names = ['artist', 'id'])

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = models.resnet50(pretrained=True).to(device)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(2048, 72).to(device)

checkpoint = torch.load("new_model(4).pt")
model.load_state_dict(checkpoint['model_state_dict'])


def identifyArtistByPhoto(pathToFile):
    picture = Image.open(pathToFile).convert('RGB')

    for tr in ts:
        picture = tr(picture)

    picture = picture.reshape(1, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        output = model(picture)
    id_ = np.argmax(output.data.cpu().numpy())
    new_df = df.loc[df['id'] == id_]
    artist = new_df['artist']
    index_list = list(artist.index)
    res_artist = artist[index_list[0]].partition('/')[0]
    return res_artist

bot.polling(none_stop=True)