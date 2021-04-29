
import numpy as np
import soundfile
import librosa.display
import random


#自适应滤波-最小均方差准则
def LMS(xn, dn, M, mu, itr):
    """
    使用LMS自适应滤波
    :param xn:输入的信号序列
    :param dn:所期望的响应序列
    :param M:滤波器的阶数
    :param mu:收敛因子(步长)
    :param itr:迭代次数
    :return:
    """
    en = np.zeros(itr)  # 误差序列,en(k)表示第k次迭代时预期输出与实际输入的误差
    W = np.zeros((M, itr))  # 每一行代表一个加权参量,每一列代表-次迭代,初始为0
    # 迭代计算
    for k in range(M, itr):
        x = xn[k:k - M:-1]
        y = np.matmul(W[:, k - 1], x)
        en[k] = dn[k] - y
        W[:, k] = W[:, k - 1] + 2 * mu * en[k] * x
    # 求最优输出序列
    yn = np.inf * np.ones(len(xn))    #初值为无穷大是绘图使用，无穷大处不会绘图
    for k in range(M, len(xn)):
        x = xn[k:k - M:-1]
        yn[k] = np.matmul(W[:, -1], x) 
    return yn, W, en


if __name__=="__main__":
  
    #参数初始化
    M=64 #滤波器的阶数
    mu=0.0001 #步长因子
    itr=5000    #迭代次数
    do_cmvn=0


    audio, sr = librosa.load("recording-20210122.wav")
    if False:
        audio -= np.mean(audio)
        audio /= np.max(np.abs(audio))

    #noise_array =  np.random.normal(0, 0.3, len(audio))

    (result,w,en)=LMS(audio,audio,M,mu,itr)

    if False:
        print(result[:100])
        result=result[100:]
        #result[np.isinf(result)] = 0.000     #替换inf
        result -= np.mean(result)
        result /= np.max(np.abs(result))
    soundfile.write("result_lms.wav", result, sr)

    '''
    plt.figure(1)
    librosa.display.waveplot(audio, sr=sr)
    plt.figure(2)
    librosa.display.waveplot(result, sr=sr)
    plt.show()

'''

    

