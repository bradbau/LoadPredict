
import os
import pandas as pd
import numpy as np

def GenerateTrainingData(inputFilePath, outputFilePath, seriesLength):
    #根据seriesLength确定输出的训练数据的维度，表示用多长的序列去预测
    if(os.path.isdir(inputFilePath)):
        files= os.listdir(inputFilePath)
        for fileItem in files:
            inputFileFullName=inputFilePath+"\\"+fileItem
            df=pd.read_table(inputFileFullName,sep=',', header=None)
            print("reading "+fileItem)
            print(df.shape)
            #print(df[0])
            dataArray=df.values.T
            outDF=pd.DataFrame(None)
            print("numpy array shape ", dataArray.shape)
            for i in range(dataArray.shape[0]):
                for j in range(dataArray.shape[1]-2):
                    outDF.append([dataArray[i,j], dataArray[i,j+1], dataArray[i,j+2]])
            print(outDF.shape)
            outDF.to_csv(inputFilePath[0:-3]+"seriesLength"+string(seriesLength)+"\\"+fileaitem)


    

if __name__=="__main__":
    #test
    GenerateTrainingData(r"D:\energy_reuse_in_water_cooling_system\conference_trans\LoadPredict\inputdata\raw",r"D:\energy_reuse_in_water_cooling_system\conference_trans\LoadPredict\inputdata\test1",2)