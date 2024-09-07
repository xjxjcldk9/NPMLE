# NPMLE Two Step EB

這個repo主要秀出NPMLE問題在解完之後，拿後驗平均會consistent這件事

主要框架如下


$\theta_i\sim G$

$x_i|\theta_i \sim \varphi(.|\theta_i)$

$y_i=\beta f(\theta_i)+\epsilon_i$


看不到$\theta_i$，只能看到$x_i,$ $y_i$，關心的是第二階段$\beta$的估計，假設Likelihood、二階段函數形式已知

抽象的步驟如下：

1. 將Likelihood矩陣丟進NPMLE得出每個grid point的weight
2. 算出第二階段會用到的posterior mean
3. 跑OLS


我們想要驗證這個方法得出的OLS結果會不會consistent

## 主要需求



### 適應不同方法測試
因為整個流程很固定，只是方法上有一些地方需要測試，常常需要將方法拿出來比較

比較簡單的作法是寫一個函數，用不同的參數來指定要用什麼方法來解

不知道要不要弄成class感覺很麻煩

而且ALM的資料生成沒那麼單純ＱＱ，有沒有必要搞那個general

1. 解NPMLE有兩個方法
- Optimal Transport
- Augmented Lagrange Multiplier

2. 我會需要測試不同的likelihood，不同的二階段函數形式

3. 維度也會不一樣，需要注意矩陣乘法的相容性

### consistenency

要能夠比較不同n的時候跑出來B個beta的結果

感覺這個功能才需要被寫成套件，但我還沒有能力去設計


## Requirements

因為ALM需要跑matlab，以現在行資的版本只能跑python3.9

挺蠢的，因為很可能之後又會改，不管是方法還是matlab版本，所以ㄏㄏ