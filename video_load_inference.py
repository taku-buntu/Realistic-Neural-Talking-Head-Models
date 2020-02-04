import torch
import cv2
from matplotlib import pyplot as plt

from loss.loss_discriminator import *
from loss.loss_generator import *
from network.blocks import *
from network.model import *
from webcam_demo.webcam_extraction_conversion import *

"""Init"""

#Paths
path_to_model_weights = 'model_weights.tar'
# path_to_model_weights = 'finetuned_model.tar'
path_to_embedding = 'e_hat_video.tar'

device = torch.device("cuda:0")
cpu = torch.device("cpu")

checkpoint = torch.load(path_to_model_weights, map_location=cpu) 
e_hat = torch.load(path_to_embedding, map_location=cpu)
e_hat = e_hat['e_hat'].to(device)

G = Generator(256, finetuning=True, e_finetuning=e_hat)
G.eval()

"""Training Init"""
G.load_state_dict(checkpoint['G_state_dict'])
G.to(device)
G.finetuning_init()


"""Main"""
print('PRESS Q TO EXIT')
cap = cv2.VideoCapture("webcam_video_lab.mov")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# fake = cv2.VideoWriter("fake.avi", fourcc, fps, (width, height))
# me = cv2.VideoWriter("me.avi", fourcc, fps, (width, height))
# landmark = cv2.VideoWriter("landmark.avi", fourcc, fps, (width, height))
result = cv2.VideoWriter("result.mp4", fourcc, fps, (int(width*3), height))

with torch.no_grad():
    i = 0
    while True:
        x, g_y = generate_landmarks(cap=cap, device=device, pad=50)

        if x is None:
            break
        g_y = g_y.unsqueeze(0)
        x = x.unsqueeze(0)


        #forward
        # Calculate average encoding vector for video
        #f_lm_compact = f_lm.view(-1, f_lm.shape[-4], f_lm.shape[-3], f_lm.shape[-2], f_lm.shape[-1]) #BxK,2,3,224,224
        #train G

        x_hat = G(g_y, e_hat)

        plt.clf()
        out1 = x_hat.transpose(1,3)[0]/255
        #for img_no in range(1,x_hat.shape[0]):
        #    out1 = torch.cat((out1, x_hat.transpose(1,3)[img_no]), dim = 1)
        out1 = out1.to(cpu).numpy()
        #plt.imshow(out1)
        #plt.show()
        
        #plt.clf()
        out2 = x.transpose(1,3)[0]/255
        #for img_no in range(1,x.shape[0]):
        #    out2 = torch.cat((out2, x.transpose(1,3)[img_no]), dim = 1)
        out2 = out2.to(cpu).numpy()
        #plt.imshow(out2)
        #plt.show()

        #plt.clf()
        out3 = g_y.transpose(1,3)[0]/255
        #for img_no in range(1,g_y.shape[0]):
        #    out3 = torch.cat((out3, g_y.transpose(1,3)[img_no]), dim = 1)
        out3 = out3.to(cpu).numpy()
        #plt.imshow(out3)
        #plt.show()
        
        # cv2.imshow('fake', cv2.cvtColor(out1, cv2.COLOR_BGR2RGB))
        # cv2.imshow('me', cv2.cvtColor(out2, cv2.COLOR_BGR2RGB))
        # cv2.imshow('ladnmark', cv2.cvtColor(out3, cv2.COLOR_BGR2RGB))
        
        result_img = cv2.hconcat([out1, out2, out3])
        # cv2.imshow("result", cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        cv2.imshow("result", cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        # result.write(cv2.cvtColor((result_img*255).astype('uint8'), cv2.COLOR_BGR2RGB))

        print(f"{i+1}/{frame_count}")
        i += 1
        if i == frame_count-10:
            break

        if cv2.waitKey(1) == ord('q'):
            break
cap.release()
# fake.release()
# me.release()
# landmark.release()
result.release()
cv2.destroyAllWindows()
