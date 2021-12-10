# DinoAI-API

# **getData**
---

<!--  樣板
**Inputs**:
**Outputs**:
**Parameters**:
**Method**:
**Example**:
-->
 
## getRoi(img_path)

**Inputs**: 

    任一種類型的圖檔 Ex: jpg、png.

**Outputs**: 

    Int: x, y ,w, h

**Parameters**: 

    x: 擷取位置的x軸起始位置
    y: 擷取位置的y軸起始位置
    w: 擷取位置的寬度
    h: 擷取位置的高度

**Method**:

    將全屏截圖當成Inputs，框出需要擷取的範圍回傳擷取起始位置和寬度、高度。

**Example**:
```python=
getRoi('test.jpg')
```

設定圖片的路徑名稱並執行程式，開啟新視窗框出需要擷取的範圍，按下enter確認，按下c重新選取，再次按下enter儲存並關閉視窗。
![](https://i.imgur.com/8NDfJDp.png)

## take_screenshot( ss_id, key )

**Inputs**:

    UUID4: ss_id 
    String: key

**Outputs**:

    png images

**Parameters**:
 
    ss_id: 採用UUID4的隨機編號
    key: 按鍵名稱(up、down、enter)

**Method**:

    當玩家按下相對應的按鍵時，擷取畫面並加上編號(key+ss_id+count)存取在./images/資料夾裡

**Example**:
```python=
if keyboard.is_pressed(keyboard.KEY_UP):  # If 'up' key is pressed
    take_screenshot(ss_id, "up")
    time.sleep(0.01)
```
![](https://i.imgur.com/6BkOAPX.png)



# **train_model**
---
## get_images_and_labels( images )

**Inputs**:
 
    png images
**Outputs**:
 
    Array: arr_images, arr_labels
**Parameters**:

    images: ./images/*.png(images資料夾裡的所有圖片)
    arr_images: 將圖片轉化成陣列的形式儲存
    arr_labels: 將label轉化成陣列的形式儲存
**Method**:

    1. 透過圖片檔名，取得label資訊
    2. 將圖片轉成灰階80x75
    3. 將圖片和label都儲存為陣列
**Example**:
```python=
images = glob.glob("./images/*.png")
X, Y = get_images_and_labels(images)
```

## onehot_labels( labels )

**Inputs**:
 
    Array: labels

**Outputs**:

    Array: onehot_labels
    
**Parameters**:

    labels: 陣列裡的label資料
    onehot_labels: 已轉換成onehot的label

**Method**:

    將陣列裡的label資料全部轉換成onehot的形式

**Example**:
```python=
Y_labels = onehot_labels(Y_labels)
```

## get_CNN_model()

**Inputs**:

    N/A

**Outputs**:

    Sequential: model

**Parameters**:

    model: Sequential代表著線性疊加，可以隨著需求的不同，增加或減少部分的layer

**Method**:

![](https://i.imgur.com/n8OtA4q.png)
![](https://i.imgur.com/q2mVtFL.png)

**Example**:

* CNN模型內部
```python=
model = Sequential()
model.add(Conv2D(16, kernel_size=(5, 5), activation="relu", input_shape=(width, height, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=(5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))  # For regularization
model.add(Dense(3, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
```
* 使用
```python=
train_X, test_X, train_y, test_y = train_test_split(images, labels, test_size=0.1, random_state=10)  # Split the dataset
model = get_CNN_model()
history = model.fit(train_X, train_y, epochs=10, batch_size=64)
```

## plot_data()

**Inputs**:

    N/A

**Outputs**:

    N/A

**Parameters**:

    N/A
    
**Method**:

	將images資料夾內的各個種類的圖片彙整起來，並輸出成圖表
	將images資料夾的圖片按比例分成**訓練用**與**測試用**，並輸出成圖表

**Example**:
```python=
plot_data()
```
![](https://i.imgur.com/hbmbary.png)
![](https://i.imgur.com/vXJOLRy.png)



## plot_accuracy_and_loss()

**Inputs**:

    N/A

**Outputs**:

    N/A

**Parameters**:

    N/A
    
**Method**:

    將訓練的**accuracy**輸出成圖表
    將訓練的**loss**輸出成圖表

**Example**:
```python=
plot_accuracy_and_loss()
```
![](https://i.imgur.com/wJ22nZj.png)
![](https://i.imgur.com/LeqVvfO.png)


## plot_confusion_matrix()

**Inputs**:

    N/A

**Outputs**:

    N/A

**Parameters**:

    N/A  

**Method**:

    輸出CNN模型的**confusion matrix**(混淆矩陣)

**Example**:
```python=
plot_confusion_matrix()
```
![](https://i.imgur.com/Y450KL7.png)


# **play_game**
---
## temp

**Inputs**:

**Outputs**:

**Parameters**:

**Method**:

**Example**:
