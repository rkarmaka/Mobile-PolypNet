import sys
sys.path.insert(0, './utils')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
import pandas as pd
import models
import func
import load_images
from datetime import datetime
import numpy as np

########################################################################################################################
np.random.seed(1)
epochs = 500
batch_size = 8

img_size=224
print('Loading data...')
X_train, y_train = load_images.load_images(img_size)
print('Data loading complete...')
print(y_train.max())
########################################################################################################################
model_mobile_polypNet=models.mobile_polypNet_maxPool(img_size=img_size)
print(model_mobile_polypNet.summary())
model_mobile_polypNet.compile(optimizer=Adam(learning_rate=0.0001),
                   loss=func.dice_coef_loss,
                   metrics=['acc',func.dice_coef])
earlystop=EarlyStopping(monitor='val_dice_coef',patience=25,mode='max')
checkpoint_path='results/model_checkpoint_mobile_polypNet_maxpool/Checkpoint_best'
log_dir = 'logs/mobile_polypNet_maxpool' + datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
checkpoint=ModelCheckpoint(filepath=checkpoint_path,
                           monitor='val_dice_coef',
                           save_best_only=True,
                           save_weights_only=True,
                           mode='max')

history_mobile_polypNet=model_mobile_polypNet.fit(X_train,y_train,epochs=epochs,batch_size=batch_size,
                            verbose=1,validation_split=0.1,
                            callbacks=[earlystop,checkpoint,tensorboard_callback])
########################################################################################################################
history_model_mobile_polypNet_df=pd.DataFrame(history_mobile_polypNet.history)
history_model_mobile_polypNet_df.to_csv('results/model_mobile_polypNet_maxpool.csv')
########################################################################################################################
# Performance measure