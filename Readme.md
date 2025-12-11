Dataset URL: https://www.kaggle.com/datasets/eron93br/obd2data
License(s): CC-BY-SA-4.0
Downloading obd2data.zip to /content
  0% 0.00/1.41M [00:00<?, ?B/s]
100% 1.41M/1.41M [00:00<00:00, 986MB/s]
Downloaded and moved dataset to /content/obd2data
Found 39 CSV files. First few: ['/content/obd2data/live1.csv', '/content/obd2data/live10.csv', '/content/obd2data/live11.csv', '/content/obd2data/live12.csv', '/content/obd2data/live13.csv']
Concatenated dataframe shape: (97281, 29)
Using sensor columns: ['ENGINE_RPM ()', 'ENGINE_RUN_TINE ()', 'VEHICLE_SPEED ()', 'THROTTLE ()', 'ENGINE_LOAD ()', 'COOLANT_TEMPERATURE ()', 'INTAKE_MANIFOLD_PRESSURE ()', 'INTAKE_AIR_TEMP ()', 'TIMING_ADVANCE ()']
Applying per-unit linear interpolation + ffill/bfill for sensor cols...
Interpolation done. Remaining NaNs (total): 0
Building auto threshold rules (percentile-based)...
Auto rules sample (first 8): [('ENGINE_RPM ()', 'gt', 1932.8, 'severe'), ('ENGINE_RPM ()', 'gt', 1539.0, 'minor'), ('ENGINE_RUN_TINE ()', 'gt', 2225.0, 'severe'), ('ENGINE_RUN_TINE ()', 'gt', 1950.75, 'minor'), ('VEHICLE_SPEED ()', 'gt', 56.0, 'severe'), ('VEHICLE_SPEED ()', 'gt', 38.0, 'minor'), ('THROTTLE ()', 'gt', 91.764709, 'severe'), ('THROTTLE ()', 'gt', 68.235291, 'minor')]
Applying auto-rules to create fault_label...
Label distribution: {'healthy': 69879, 'minor': 20844, 'severe': 6558}
Window-level shape: (19088, 50, 9) label counts: Counter({np.str_('healthy'): 14572, np.str_('minor'): 3560, np.str_('severe'): 956})
Classifier classes: ['healthy', 'minor', 'severe']
Training RandomForest classifier...
Classifier report:
               precision    recall  f1-score   support

     healthy       0.99      0.99      0.99      2915
       minor       0.96      0.96      0.96       712
      severe       0.97      0.93      0.95       191

    accuracy                           0.98      3818
   macro avg       0.97      0.96      0.97      3818
weighted avg       0.98      0.98      0.98      3818

Severity (weighted) stats: 0.0 7.5 0.8341929484492876

/tmp/ipython-input-2960135633.py:289: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df["severity"].ffill(inplace=True); df["severity"].bfill(inplace=True); df["severity"].fillna(0.0, inplace=True)

Row-level severity stats: 0.0 7.5 0.8236693187775619
Synthetic RUL overall stats: {'count': 97281.0, 'mean': 871.3572232519354, 'std': 259.8362096941857, 'min': 0.0, '25%': 903.9301310043668, '50%': 1000.0, '75%': 1000.0, 'max': 1000.0}
RUL dataset shapes: (19088, 50, 9) (19088,)
Train/Val/Test shapes: (13361, 50, 9) (2863, 50, 9) (2864, 50, 9)

Model: "functional"

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer (InputLayer)        │ (None, 50, 9)          │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv1d (Conv1D)                 │ (None, 50, 32)         │           896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization             │ (None, 50, 32)         │           128 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling1d (MaxPooling1D)    │ (None, 25, 32)         │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv1d_1 (Conv1D)               │ (None, 25, 64)         │         6,208 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_1           │ (None, 25, 64)         │           256 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling1d_1 (MaxPooling1D)  │ (None, 12, 64)         │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 12, 64)         │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ lstm (LSTM)                     │ (None, 64)             │        33,024 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 1)              │            65 │
└─────────────────────────────────┴────────────────────────┴───────────────┘

 Total params: 40,577 (158.50 KB)

 Trainable params: 40,385 (157.75 KB)

 Non-trainable params: 192 (768.00 B)

Epoch 1/40
418/418 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.2952 - mae: 0.3983
Epoch 1: val_loss improved from inf to 0.03794, saving model to best_rul_model_final.h5

WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 

418/418 ━━━━━━━━━━━━━━━━━━━━ 10s 10ms/step - loss: 0.2948 - mae: 0.3979 - val_loss: 0.0379 - val_mae: 0.1303
Epoch 2/40
415/418 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - loss: 0.0528 - mae: 0.1658
Epoch 2: val_loss improved from 0.03794 to 0.03396, saving model to best_rul_model_final.h5

WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 

418/418 ━━━━━━━━━━━━━━━━━━━━ 5s 11ms/step - loss: 0.0528 - mae: 0.1657 - val_loss: 0.0340 - val_mae: 0.1222
Epoch 3/40
416/418 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - loss: 0.0415 - mae: 0.1430
Epoch 3: val_loss improved from 0.03396 to 0.03062, saving model to best_rul_model_final.h5

WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 

418/418 ━━━━━━━━━━━━━━━━━━━━ 4s 9ms/step - loss: 0.0415 - mae: 0.1430 - val_loss: 0.0306 - val_mae: 0.1146
Epoch 4/40
418/418 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0372 - mae: 0.1321
Epoch 4: val_loss improved from 0.03062 to 0.02727, saving model to best_rul_model_final.h5

WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 

418/418 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step - loss: 0.0372 - mae: 0.1321 - val_loss: 0.0273 - val_mae: 0.1042
Epoch 5/40
416/418 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0337 - mae: 0.1236
Epoch 5: val_loss improved from 0.02727 to 0.02620, saving model to best_rul_model_final.h5

WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 

418/418 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step - loss: 0.0336 - mae: 0.1236 - val_loss: 0.0262 - val_mae: 0.1024
Epoch 6/40
415/418 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - loss: 0.0311 - mae: 0.1179
Epoch 6: val_loss improved from 0.02620 to 0.02496, saving model to best_rul_model_final.h5

WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 

418/418 ━━━━━━━━━━━━━━━━━━━━ 4s 10ms/step - loss: 0.0311 - mae: 0.1179 - val_loss: 0.0250 - val_mae: 0.0987
Epoch 7/40
413/418 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0288 - mae: 0.1125
Epoch 7: val_loss improved from 0.02496 to 0.02403, saving model to best_rul_model_final.h5

WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 

418/418 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step - loss: 0.0288 - mae: 0.1125 - val_loss: 0.0240 - val_mae: 0.0973
Epoch 8/40
416/418 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0270 - mae: 0.1074
Epoch 8: val_loss improved from 0.02403 to 0.02282, saving model to best_rul_model_final.h5

WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 

418/418 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step - loss: 0.0270 - mae: 0.1074 - val_loss: 0.0228 - val_mae: 0.0927
Epoch 9/40
416/418 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - loss: 0.0256 - mae: 0.1035
Epoch 9: val_loss improved from 0.02282 to 0.02128, saving model to best_rul_model_final.h5

WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 

418/418 ━━━━━━━━━━━━━━━━━━━━ 4s 10ms/step - loss: 0.0256 - mae: 0.1035 - val_loss: 0.0213 - val_mae: 0.0869
Epoch 10/40
416/418 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0249 - mae: 0.1014
Epoch 10: val_loss did not improve from 0.02128
418/418 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step - loss: 0.0249 - mae: 0.1014 - val_loss: 0.0221 - val_mae: 0.0901
Epoch 11/40
415/418 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0237 - mae: 0.0982
Epoch 11: val_loss improved from 0.02128 to 0.02068, saving model to best_rul_model_final.h5

WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 

418/418 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step - loss: 0.0237 - mae: 0.0982 - val_loss: 0.0207 - val_mae: 0.0867
Epoch 12/40
416/418 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0227 - mae: 0.0948
Epoch 12: val_loss improved from 0.02068 to 0.02033, saving model to best_rul_model_final.h5

WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 

418/418 ━━━━━━━━━━━━━━━━━━━━ 4s 9ms/step - loss: 0.0227 - mae: 0.0948 - val_loss: 0.0203 - val_mae: 0.0860
Epoch 13/40
418/418 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0221 - mae: 0.0934
Epoch 13: val_loss improved from 0.02033 to 0.01993, saving model to best_rul_model_final.h5

WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 

418/418 ━━━━━━━━━━━━━━━━━━━━ 5s 8ms/step - loss: 0.0221 - mae: 0.0934 - val_loss: 0.0199 - val_mae: 0.0828
Epoch 14/40
417/418 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0216 - mae: 0.0918
Epoch 14: val_loss improved from 0.01993 to 0.01887, saving model to best_rul_model_final.h5

WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 

418/418 ━━━━━━━━━━━━━━━━━━━━ 4s 8ms/step - loss: 0.0216 - mae: 0.0918 - val_loss: 0.0189 - val_mae: 0.0801
Epoch 15/40
418/418 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - loss: 0.0210 - mae: 0.0905
Epoch 15: val_loss did not improve from 0.01887
418/418 ━━━━━━━━━━━━━━━━━━━━ 5s 11ms/step - loss: 0.0210 - mae: 0.0905 - val_loss: 0.0193 - val_mae: 0.0802
Epoch 16/40
415/418 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0204 - mae: 0.0874
Epoch 16: val_loss improved from 0.01887 to 0.01754, saving model to best_rul_model_final.h5

WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 

418/418 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step - loss: 0.0204 - mae: 0.0874 - val_loss: 0.0175 - val_mae: 0.0754
Epoch 17/40
416/418 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0197 - mae: 0.0865
Epoch 17: val_loss improved from 0.01754 to 0.01737, saving model to best_rul_model_final.h5

WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 

418/418 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step - loss: 0.0197 - mae: 0.0865 - val_loss: 0.0174 - val_mae: 0.0740
Epoch 18/40
414/418 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0195 - mae: 0.0857
Epoch 18: val_loss improved from 0.01737 to 0.01725, saving model to best_rul_model_final.h5

WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 

418/418 ━━━━━━━━━━━━━━━━━━━━ 4s 9ms/step - loss: 0.0195 - mae: 0.0857 - val_loss: 0.0173 - val_mae: 0.0738
Epoch 19/40
418/418 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0191 - mae: 0.0842
Epoch 19: val_loss did not improve from 0.01725
418/418 ━━━━━━━━━━━━━━━━━━━━ 4s 9ms/step - loss: 0.0191 - mae: 0.0842 - val_loss: 0.0173 - val_mae: 0.0737
Epoch 20/40
418/418 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0186 - mae: 0.0828
Epoch 20: val_loss improved from 0.01725 to 0.01649, saving model to best_rul_model_final.h5

WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 

418/418 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step - loss: 0.0186 - mae: 0.0828 - val_loss: 0.0165 - val_mae: 0.0723
Epoch 21/40
415/418 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0179 - mae: 0.0806
Epoch 21: val_loss improved from 0.01649 to 0.01623, saving model to best_rul_model_final.h5

WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 

418/418 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step - loss: 0.0179 - mae: 0.0806 - val_loss: 0.0162 - val_mae: 0.0714
Epoch 22/40
412/418 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - loss: 0.0176 - mae: 0.0803
Epoch 22: val_loss did not improve from 0.01623
418/418 ━━━━━━━━━━━━━━━━━━━━ 4s 10ms/step - loss: 0.0176 - mae: 0.0803 - val_loss: 0.0165 - val_mae: 0.0737
Epoch 23/40
411/418 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0172 - mae: 0.0791
Epoch 23: val_loss improved from 0.01623 to 0.01623, saving model to best_rul_model_final.h5

WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 

418/418 ━━━━━━━━━━━━━━━━━━━━ 4s 9ms/step - loss: 0.0172 - mae: 0.0791 - val_loss: 0.0162 - val_mae: 0.0702
Epoch 24/40
414/418 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0172 - mae: 0.0787
Epoch 24: val_loss improved from 0.01623 to 0.01598, saving model to best_rul_model_final.h5

WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 

418/418 ━━━━━━━━━━━━━━━━━━━━ 4s 8ms/step - loss: 0.0172 - mae: 0.0787 - val_loss: 0.0160 - val_mae: 0.0707
Epoch 25/40
417/418 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - loss: 0.0167 - mae: 0.0775
Epoch 25: val_loss improved from 0.01598 to 0.01560, saving model to best_rul_model_final.h5

WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 

418/418 ━━━━━━━━━━━━━━━━━━━━ 4s 10ms/step - loss: 0.0167 - mae: 0.0775 - val_loss: 0.0156 - val_mae: 0.0695
Epoch 26/40
412/418 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0163 - mae: 0.0766
Epoch 26: val_loss improved from 0.01560 to 0.01541, saving model to best_rul_model_final.h5

WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 

418/418 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step - loss: 0.0163 - mae: 0.0765 - val_loss: 0.0154 - val_mae: 0.0682
Epoch 27/40
417/418 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0161 - mae: 0.0754
Epoch 27: val_loss improved from 0.01541 to 0.01527, saving model to best_rul_model_final.h5

WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 

418/418 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step - loss: 0.0161 - mae: 0.0754 - val_loss: 0.0153 - val_mae: 0.0673
Epoch 28/40
414/418 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0158 - mae: 0.0750
Epoch 28: val_loss improved from 0.01527 to 0.01507, saving model to best_rul_model_final.h5

WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 

418/418 ━━━━━━━━━━━━━━━━━━━━ 4s 9ms/step - loss: 0.0158 - mae: 0.0750 - val_loss: 0.0151 - val_mae: 0.0671
Epoch 29/40
416/418 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0157 - mae: 0.0741
Epoch 29: val_loss did not improve from 0.01507
418/418 ━━━━━━━━━━━━━━━━━━━━ 4s 9ms/step - loss: 0.0157 - mae: 0.0741 - val_loss: 0.0152 - val_mae: 0.0664
Epoch 30/40
418/418 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0145 - mae: 0.0711
Epoch 30: val_loss improved from 0.01507 to 0.01471, saving model to best_rul_model_final.h5

WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 

418/418 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step - loss: 0.0145 - mae: 0.0711 - val_loss: 0.0147 - val_mae: 0.0654
Epoch 31/40
415/418 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0152 - mae: 0.0729
Epoch 31: val_loss improved from 0.01471 to 0.01460, saving model to best_rul_model_final.h5

WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 

418/418 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step - loss: 0.0151 - mae: 0.0729 - val_loss: 0.0146 - val_mae: 0.0657
Epoch 32/40
415/418 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - loss: 0.0151 - mae: 0.0729
Epoch 32: val_loss improved from 0.01460 to 0.01401, saving model to best_rul_model_final.h5

WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 

418/418 ━━━━━━━━━━━━━━━━━━━━ 4s 10ms/step - loss: 0.0151 - mae: 0.0729 - val_loss: 0.0140 - val_mae: 0.0651
Epoch 33/40
418/418 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0145 - mae: 0.0710
Epoch 33: val_loss improved from 0.01401 to 0.01397, saving model to best_rul_model_final.h5

WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 

418/418 ━━━━━━━━━━━━━━━━━━━━ 4s 8ms/step - loss: 0.0145 - mae: 0.0710 - val_loss: 0.0140 - val_mae: 0.0635
Epoch 34/40
413/418 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0142 - mae: 0.0701
Epoch 34: val_loss did not improve from 0.01397
418/418 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step - loss: 0.0142 - mae: 0.0701 - val_loss: 0.0141 - val_mae: 0.0627
Epoch 35/40
418/418 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - loss: 0.0142 - mae: 0.0699
Epoch 35: val_loss did not improve from 0.01397
418/418 ━━━━━━━━━━━━━━━━━━━━ 4s 10ms/step - loss: 0.0142 - mae: 0.0699 - val_loss: 0.0143 - val_mae: 0.0645
Epoch 36/40
414/418 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0141 - mae: 0.0696
Epoch 36: val_loss improved from 0.01397 to 0.01372, saving model to best_rul_model_final.h5

WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 

418/418 ━━━━━━━━━━━━━━━━━━━━ 4s 8ms/step - loss: 0.0141 - mae: 0.0696 - val_loss: 0.0137 - val_mae: 0.0626
Epoch 37/40
418/418 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0140 - mae: 0.0691
Epoch 37: val_loss did not improve from 0.01372
418/418 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step - loss: 0.0140 - mae: 0.0691 - val_loss: 0.0140 - val_mae: 0.0641
Epoch 38/40
417/418 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0134 - mae: 0.0677
Epoch 38: val_loss did not improve from 0.01372
418/418 ━━━━━━━━━━━━━━━━━━━━ 4s 9ms/step - loss: 0.0134 - mae: 0.0677 - val_loss: 0.0139 - val_mae: 0.0620
Epoch 39/40
414/418 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0130 - mae: 0.0665
Epoch 39: val_loss improved from 0.01372 to 0.01338, saving model to best_rul_model_final.h5

WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 

418/418 ━━━━━━━━━━━━━━━━━━━━ 4s 9ms/step - loss: 0.0130 - mae: 0.0665 - val_loss: 0.0134 - val_mae: 0.0616
Epoch 40/40
415/418 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0130 - mae: 0.0667
Epoch 40: val_loss improved from 0.01338 to 0.01334, saving model to best_rul_model_final.h5

WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 

418/418 ━━━━━━━━━━━━━━━━━━━━ 5s 8ms/step - loss: 0.0130 - mae: 0.0667 - val_loss: 0.0133 - val_mae: 0.0610
Restoring model weights from the end of the best epoch: 40.
90/90 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step
Final Test RMSE: 108.453, R2: 0.822

WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 

Saved: final_rul_model_eron93br.h5, best_rul_model_final.h5, fault_clf_eron93br.pkl, scaler_rul_eron93br.pkl
<img width="994" height="374" alt="image" src="https://github.com/user-attachments/assets/eefe8c2a-93a0-4c0b-8dc6-ba38f92af92f" />



