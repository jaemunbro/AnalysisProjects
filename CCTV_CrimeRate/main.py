import pandas as pd
import glob
import os
import numpy as np

def astype_df_columns_to_numeric(df, column_list):
    df[column_list] = df[column_list].astype(int)

input_file = '..\Data' #데이터 파일 경로
# print(input_file) #경로 확인용

allFile_list = glob.glob(os.path.join(input_file, '*')) #데이터 파일 경로 저장
# print(allFile_list) #데이터 파일 경로 확인
allData = [] #csv저장 변수

# [0] : crime_2014 [1] : crime_2015 [2] : crime_2016 [3] : seoul_cctv
for file in allFile_list : #csv 저장
    df = pd.read_csv(file, encoding = 'CP949')
    allData.append(df)

main_df = pd.DataFrame()#병합용 데이터 프레임

#데이터 프레임 열 이름 및 인덱스 정리
for n in range(0,len(allData)) :
    if(n == 3) :
        #열이름 변경
        allData[n].columns = ['자치구','합계','2013년','2014년','2015년','2016년']
        #필요없는 열 삭제
        del allData[n]['합계']
        #CCTV누적량 계산(기존 값은 해당 년도에 증가한 CCTV의 수)
        allData[n]['2014년'] = allData[n]['2013년'] + allData[n]['2014년']
        allData[n]['2015년'] = allData[n]['2014년'] + allData[n]['2015년']
        allData[n]['2016년'] = allData[n]['2015년'] + allData[n]['2016년']
    else :
        #열이름 변경
        allData[n].columns = ['년도','자치구','총범죄발생','총범죄검거','살인발생','살인검거','강도발생','강도검거','강제추행발생','강제추행검거',
             '절도발생','절도검거','폭력발생','폭력검거']
        # 의미 없는 행 삭제 인덱스 재정리
        allData[n] = allData[n].drop([0,1])
        allData[n] = allData[n].reset_index(drop = True)
        #main_df에 데이터 추가
        main_df = main_df.append(allData[n])

#CCTV현황 열 추가
main_df['CCTV현황'] = np.nan
#열과 인덱스 재정렬
main_df = pd.DataFrame(main_df,columns = ['자치구','년도','CCTV현황','총범죄발생','총범죄검거','살인발생','살인검거','강도발생','강도검거',
                                          '강제추행발생','강제추행검거','절도발생','절도검거','폭력발생','폭력검거'])
main_df = main_df.reset_index(drop = True)

#main_df에 CCTV수 추가
for r_main in range(0, len(main_df.axes[0])) :
    for r_cctv in range(0, len(allData[3].axes[0])) :
        if(main_df.iloc[r_main]['자치구'] == allData[3].iloc[r_cctv]['자치구']) :
            if(main_df.iloc[r_main]['년도'] == '2014') :
                main_df.set_value(r_main,'CCTV현황',allData[3].iloc[r_cctv]['2014년'])
            elif(main_df.iloc[r_main]['년도'] == '2015') :
                main_df.set_value(r_main, 'CCTV현황', allData[3].iloc[r_cctv]['2015년'])
            elif (main_df.iloc[r_main]['년도'] == '2016'):
                main_df.set_value(r_main, 'CCTV현황', allData[3].iloc[r_cctv]['2016년'])

#튜플들 쉼표 제거
main_df.replace(regex=True,inplace=True,to_replace=r'\W',value=r'')

#숫자형으로 형변환
astype_df_columns_to_numeric(main_df, ['년도','CCTV현황','총범죄발생','총범죄검거','살인발생','살인검거','강도발생','강도검거',
                                          '강제추행발생','강제추행검거','절도발생','절도검거','폭력발생','폭력검거'])

for x in range(0, len(main_df.columns)) :
    print(type(main_df.iloc[0][x]))
print(main_df)