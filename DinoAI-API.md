# DinoAI-API

## Environment

- Win10
- Dino(google 小恐龍遊戲)
- Python 3.9.0
- tensorflow keras
- opencv

---

# **getData**

<!--  樣板
**Inputs**:
**Outputs**:
**Arguments**:
**Method**:
**Example**:
-->

## getRoi(img_path)

**Inputs**:

    任一種類型的圖檔 Ex: jpg、png.

**Outputs**:

    Int: x, y ,w, h

**Arguments**:

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

設定圖片的路徑名稱並執行程式，開啟新視窗框出需要擷取的範圍，按下 enter 確認，按下 c 重新選取，再次按下 enter 儲存並關閉視窗。
![](https://i.imgur.com/LHAGu8S.png)




## take_screenshot( ss_id, key )

**Inputs**:

    UUID4: ss_id
    String: key

**Outputs**:

    png image file

**Arguments**:

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

---

# **train_model**

## get_images_and_labels( images )

**Inputs**:

    png images

**Outputs**:

    Array: arr_images, arr_labels

**Arguments**:

    images: images資料夾裡的所有圖片
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

**Arguments**:

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

**Arguments**:

    model: Sequential代表著線性疊加，可以隨著需求的不同，增加或減少部分的layer

**Method**:

![](https://i.imgur.com/n8OtA4q.png)
![](https://i.imgur.com/q2mVtFL.png)
![](https://i.imgur.com/Cl3UtU8.png)


**Example**:

- CNN 模型內部

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
- 模型內部說明

    Squential: 代表模型是一層層疊加的
    
    model.add: 給模型增加一層
    
        Conv2D: 2d的convolution: 16個filiters, 大小5x5, 啟用relu, 
                定義input_shape,stride預設1。
                
        MaxPooling2D: 2x2長寬變為原來的一半。
        
        Flatten: 銜接CNN層與全連接層，因為FC層需要一維的輸入，
                 常見其他方式還有Global average pooling。
                 
        Dense: 這層FC會有128個Neurons。
        
        Dropout: 每一次的迭代 (epoch)皆以一定的機率丟棄隱藏層神經元，而被丟棄的神經元不會傳遞訊息。
        
    [Dropout說明](https://medium.com/%E6%89%8B%E5%AF%AB%E7%AD%86%E8%A8%98/%E4%BD%BF%E7%94%A8-tensorflow-%E4%BA%86%E8%A7%A3-dropout-bf64a6785431)

    [model.compile](https://dotblogs.com.tw/greengem/2017/12/17/094023): 
    定義模型的[損失函數(loss)](https://chih-sheng-huang821.medium.com/%E6%A9%9F%E5%99%A8-%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92-%E5%9F%BA%E7%A4%8E%E4%BB%8B%E7%B4%B9-%E6%90%8D%E5%A4%B1%E5%87%BD%E6%95%B8-loss-function-2dcac5ebb6cb), 
    [優化函數(optimizer)](https://medium.com/%E9%9B%9E%E9%9B%9E%E8%88%87%E5%85%94%E5%85%94%E7%9A%84%E5%B7%A5%E7%A8%8B%E4%B8%96%E7%95%8C/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92ml-note-sgd-momentum-adagrad-adam-optimizer-f20568c968db), 
    成效衡量指標(metrics)
    
    tensorflow可使用的: [losses](https://www.tensorflow.org/api_docs/python/tf/keras/losses), 
                      [optimizers](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers) 
                      [mertrics](https://www.tensorflow.org/api_docs/python/tf/keras/metrics)
        

- 使用

```python=
train_X, test_X, train_y, test_y = train_test_split(images, labels, test_size=0.1, random_state=10)  # Split the dataset
model = get_CNN_model()
history = model.fit(train_X, train_y, epochs=10, batch_size=64)
train_accuracy = model.evaluate(train_X, train_y)
print("Train accuracy: %", train_accuracy[1] * 100)

test_accuracy = model.evaluate(test_X, test_y)
print("Test accuracy: %", test_accuracy[1] * 100)
```

## plot_data("img_path", labels)

**Inputs**:

    String: image_path
    Array: labels

**Outputs**:

    圖表

**Arguments**:

    N/A

**Method**:

    將images資料夾內的各個種類的圖片彙整起來，並輸出成圖表
    將images資料夾的圖片按比例分成訓練用與測試用，並輸出成圖表

**Example**:

```python=
plot_data()
```

![](https://i.imgur.com/hbmbary.png)
![](https://i.imgur.com/vXJOLRy.png)

## plot_accuracy_and_loss(model_acc, model_loss)

**Inputs**:

    model_acc
    model_loss

**Outputs**:

    圖表

**Arguments**:

    model_acc: 透過model.history裡獲取的accuracy
    model_loss: 透過model.history裡獲取的loss

**Method**:

    將訓練的accuracy輸出成圖表
    將訓練的loss輸出成圖表

**Example**:

```python=
plot_accuracy_and_loss()
```

![](https://i.imgur.com/wJ22nZj.png)
![](https://i.imgur.com/LeqVvfO.png)

## plot_confusion_matrix(test_images, test_labels)

**Inputs**:

    Array: test_images
    Array: test_labels

**Outputs**:

    圖表

**Arguments**:

    test_images: 測試用的圖片
    test_labels: 測試用的標籤

**Method**:

    輸出CNN模型的confusion matrix(混淆矩陣)

**Example**:

```python=
plot_confusion_matrix()
```

![](https://i.imgur.com/Y450KL7.png)

---

# **play_game**

## get_trained_data

**Inputs**:

    model.json
    weight.h5

**Outputs**:

    N/A

**Arguments**:

    model_json: CNN模型的json檔
    weight.h5: 訓練完成後的資料

**Method**:

    載入訓練模型和資料

**Example**:

```python=
model = model_from_json(open("model.json", "r").read())
model.load_weights("weights.h5")
```
[HDF5/.h5檔說明連結:](https://zh.wikipedia.org/wiki/HDF)

## grab_frame

**Inputs**:

    frame

**Outputs**:

    image

**Arguments**:

    frame: 擷取畫面的大小
    image: 透過ss_manager.grab定義創建圖片大小，是否要RGB

**Method**:

    擷取遊戲當前畫面

**Example**:

```python=
screenshot = ss_manager.grab(frame)
image = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
```
datapyte of screenshot <class 'mss.screenshot.ScreenShot'>
    


## process_images

**Inputs**:

    遊戲當前畫面

**Outputs**:

    處理過的遊戲當前畫面

**Arguments**:

    N/A

**Method**:

    遊戲畫面的處理: 灰階、resize、normalize、轉換成array

**Example**:

```python=
grey_image = image.convert("L")  # Convert RGB image to grey_scale image
a_img = np.array(grey_image.resize((width, height)))  # Resize the grey image and convert it to numpy array
img = a_img / 255  # Normalize the image array

X = np.array([img])  # Convert list X to numpy array
X = X.reshape(X.shape[0], width, height, 1)  # Reshape the X
```
使用function: current_frame
```python=
def current_frame(frame):
    cv2.namedWindow("current frame", flags=cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
    cv2.imshow("current frame", frame)
    cv2.waitKey(0)
```
    輸出結果:
![](https://i.imgur.com/LouxtV1.png)

## predict

**Inputs**:

    image

**Outputs**:

    int: result

**Arguments**:

    image: 已處理過的遊戲畫面

    result:  0 = right
             1 = down
             2 = up

**Method**:

    預測當前的畫面類別為何

**Example**:

```python=
prediction = model.predict(X)  # Get prediction by using the model
result = np.argmax(prediction)  # Convert one-hot prediction to the number
```

## output_keyboard(key)

**Inputs**:

    int: key

**Outputs**:

    鍵盤輸出

**Arguments**:

    key:  0 = right
          1 = down
          2 = up

**Method**:

    模擬鍵盤輸出

**Example**:

```python=
output_keyboard(result)
```
