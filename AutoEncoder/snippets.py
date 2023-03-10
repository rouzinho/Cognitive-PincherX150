self.encoder = torch.nn.Sequential(
            #128 * 128 * 1
            nn.Conv2d(1, 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            nn.Conv2d(4, 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            #64 * 64 * 16
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            #32*32 * 32
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(4*4*64,512),
            nn.ReLU(True),
            nn.Linear(512,256),
            nn.ReLU(True),
            nn.Linear(256,128),
            nn.ReLU(True),
            nn.Linear(128,64),
            nn.ReLU(True),
            nn.Linear(64,32),
            nn.ReLU(True),
            nn.Linear(32,16),
            nn.Tanh(),
            nn.Linear(16,8),
        )
        self.decoder = torch.nn.Sequential(
            nn.Linear(8,16),
            nn.Tanh(),
            nn.Linear(16,32),
            nn.ReLU(True),
            nn.Linear(32,64),
            nn.ReLU(True),
            nn.Linear(64,128),
            nn.ReLU(True),
            nn.Linear(128,256),
            nn.ReLU(True),
            nn.Linear(256,512),
            nn.ReLU(True),
            nn.Linear(512,4*4*64),
            nn.ReLU(True),
            nn.Unflatten(1, (64, 4, 4)),
            #8 * 8 * 64
            nn.ConvTranspose2d(64, 32, 3, stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            #16*16*128
            nn.ConvTranspose2d(16, 8, 3, stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            #32 * 32 * 64
            nn.ConvTranspose2d(8, 4, 3, stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            #64 * 64 * 32
            nn.ConvTranspose2d(4, 1, 3, stride=2,padding=1,output_padding=1),
            #Trim(),
            nn.Sigmoid(),
        )
        self.training = []
        self.test = []




#bestmight overfit

self.encoder = torch.nn.Sequential(
            #128 * 128 * 1
            nn.Conv2d(1, 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            nn.Conv2d(4, 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            #64 * 64 * 16
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            #32*32 * 32
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(2*2*512,1024),
            nn.ReLU(True),
            #nn.Linear(2048,1024),
            #nn.ReLU(True),
            nn.Linear(1024,512),
            nn.ReLU(True),
            nn.Linear(512,256),
            nn.ReLU(True),
            nn.Linear(256,128),
            nn.ReLU(True),
            nn.Linear(128,64),
            nn.ReLU(True),
            nn.Linear(64,32),
            nn.ReLU(True),
            nn.Linear(32,16),
            nn.Tanh(),
            nn.Linear(16,8),
        )
        self.decoder = torch.nn.Sequential(
            nn.Linear(8,16),
            nn.Tanh(),
            nn.Linear(16,32),
            nn.ReLU(True),
            nn.Linear(32,64),
            nn.ReLU(True),
            nn.Linear(64,128),
            nn.ReLU(True),
            nn.Linear(128,256),
            nn.ReLU(True),
            nn.Linear(256,512),
            nn.ReLU(True),
            nn.Linear(512,1024),
            nn.ReLU(True),
            #nn.Linear(1024,2048),
            #nn.ReLU(True),
            nn.Linear(1024,2*2*512),
            nn.ReLU(True),
            nn.Unflatten(1, (512,2, 2)),
            #8 * 8 * 64
            nn.ConvTranspose2d(512, 256, 3, stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 3, stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            #16*16*128
            nn.ConvTranspose2d(32, 16, 3, stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            #32 * 32 * 64
            nn.ConvTranspose2d(16, 8, 3, stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            #64 * 64 * 32
            nn.ConvTranspose2d(8, 4, 3, stride=1,padding=1,output_padding=0),
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            nn.ConvTranspose2d(4, 1, 3, stride=1,padding=1,output_padding=0),
            #Trim(),
            nn.Sigmoid(),
        )
        self.training = []
        self.test = []


#best but slow
self.encoder = torch.nn.Sequential(
            #128 * 128 * 1
            nn.Conv2d(1, 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            nn.Conv2d(4, 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            #64 * 64 * 16
            nn.Conv2d(8, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            #32*32 * 32
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(4*4*512,4096),
            nn.ReLU(True),
            nn.Linear(4096,2048),
            nn.ReLU(True),
            nn.Linear(2048,1024),
            nn.ReLU(True),
            nn.Linear(1024,512),
            nn.ReLU(True),
            nn.Linear(512,256),
            nn.ReLU(True),
            nn.Linear(256,128),
            nn.ReLU(True),
            nn.Linear(128,64),
            nn.ReLU(True),
            nn.Linear(64,32),
            nn.ReLU(True),
            nn.Linear(32,16),
            #nn.Tanh(),
            nn.ReLU(True),
            nn.Linear(16,8),
        )
        self.decoder = torch.nn.Sequential(
            nn.Linear(8,16),
            #nn.Tanh(),
            nn.ReLU(True),
            nn.Linear(16,32),
            nn.ReLU(True),
            nn.Linear(32,64),
            nn.ReLU(True),
            nn.Linear(64,128),
            nn.ReLU(True),
            nn.Linear(128,256),
            nn.ReLU(True),
            nn.Linear(256,512),
            nn.ReLU(True),
            nn.Linear(512,1024),
            nn.ReLU(True),
            nn.Linear(1024,2048),
            nn.ReLU(True),
            nn.Linear(2048,4096),
            nn.ReLU(True),
            nn.Linear(4096,4*4*512),
            nn.ReLU(True),
            nn.Unflatten(1, (512,4, 4)),
            #8 * 8 * 64
            nn.ConvTranspose2d(512, 256, 3, stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 3, stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            #16*16*128
            nn.ConvTranspose2d(32, 16, 3, stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            #32 * 32 * 64
            nn.ConvTranspose2d(16, 8, 3, stride=1,padding=1,output_padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            #64 * 64 * 32
            nn.ConvTranspose2d(8, 4, 3, stride=1,padding=1,output_padding=0),
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            nn.ConvTranspose2d(4, 1, 3, stride=1,padding=1,output_padding=0),
            #Trim(),
            nn.Sigmoid(),
        )

# good and fast
self.encoder = torch.nn.Sequential(
            #128 * 128 * 1
            nn.Conv2d(1, 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            nn.Conv2d(4, 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            #64 * 64 * 16
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            #32*32 * 32
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(4*4*256,2048),
            nn.ReLU(True),
            #nn.Linear(4096,2048),
            #nn.ReLU(True),
            nn.Linear(2048,1024),
            nn.ReLU(True),
            nn.Linear(1024,512),
            nn.ReLU(True),
            nn.Linear(512,256),
            nn.ReLU(True),
            nn.Linear(256,128),
            nn.ReLU(True),
            nn.Linear(128,64),
            nn.ReLU(True),
            nn.Linear(64,32),
            nn.ReLU(True),
            nn.Linear(32,16),
            #nn.Tanh(),
            nn.ReLU(True),
            nn.Linear(16,8),
        )
        self.decoder = torch.nn.Sequential(
            nn.Linear(8,16),
            #nn.Tanh(),
            nn.ReLU(True),
            nn.Linear(16,32),
            nn.ReLU(True),
            nn.Linear(32,64),
            nn.ReLU(True),
            nn.Linear(64,128),
            nn.ReLU(True),
            nn.Linear(128,256),
            nn.ReLU(True),
            nn.Linear(256,512),
            nn.ReLU(True),
            nn.Linear(512,1024),
            nn.ReLU(True),
            nn.Linear(1024,2048),
            nn.ReLU(True),
            #nn.Linear(2048,4096),
            #nn.ReLU(True),
            nn.Linear(2048,4*4*256),
            nn.ReLU(True),
            nn.Unflatten(1, (256,4, 4)),
            #8 * 8 * 64
            nn.ConvTranspose2d(256, 256, 3, stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 3, stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            #16*16*128
            nn.ConvTranspose2d(32, 16, 3, stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            #32 * 32 * 64
            nn.ConvTranspose2d(16, 8, 3, stride=1,padding=1,output_padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            #64 * 64 * 32
            nn.ConvTranspose2d(8, 4, 3, stride=1,padding=1,output_padding=0),
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            nn.ConvTranspose2d(4, 1, 3, stride=1,padding=1,output_padding=0),
            #Trim(),
            nn.Sigmoid(),
        )