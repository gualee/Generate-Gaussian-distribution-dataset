# Generate-Gaussian-distribution-dataset
### 透過Python來產生高斯分布的資料集

### 以下為演算法步驟：

1.輸入維度為20及sample為100，產生mean vector為0、p1為0.9及p2為0.5的covariance matrix(建立一計算covariance matrix函式)

2.透過numpy的random.multivariate_normal函式產生一個高斯分佈的資料集，並輸出儲存成csv檔案

3.將資料集以圖表方式呈現出來

欲產生兩個在高斯分佈假設下的資料集，維度皆為20，並分別產生一百個samples，這裡透過建立一計算函式分別設定ρ1為0.9、ρ2為0.5來計算出covariance matrix。而mean vector部分維度大小要和covariance matrix相同，所以設定為20個0的陣列。接下來使用numpy套件的multivariate normal函式，輸入mean vector、covariance matrix及100個samples，分別輸出dataset1及dataset2，這裡將這兩個資料集利用numpy套件的savetxt方法將資料集輸出成csv格式的檔案，format為到小數點後四位，數值之間以逗號分開(csv)。最後部分使用matplotlib.pyplot的套件來產生圖表內容，分別用藍色及紅色的點代表class1和class2的資料點，兩個圖用title表示dataset1和dataset2，建立網格資料assign成X和Y，再用contour(或使用contourf亦可)呈現等高線的情形，最後使用show()將圖片呈現出來。
