import pandas as pd
import glob
import os
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import platform
from matplotlib import font_manager, rc

#데이터 프레임 특정 열을 숫자형으로 변경
def astype_df_columns_to_numeric(df, column_list) :
    df[column_list] = df[column_list].astype(int)


#피어슨 계수가 1에 가까울수록 양의 상관관계가 높다.
#피어슨 계수가 -1에 가까울수록 음의 상관관계가 높다.
#피어슨 계수의 절댓값이 0에 가까울수록 의미가 없다.

#피어슨 계수가 가장 낮은 열의 이름과 값 반환
def cal_df_min_pearson_by(df, col_name) :
    min_value = 1;
    for n_col in range(0, len(df.columns)) :
        pearson_value = df[col_name].corr(df[df.columns[n_col]])
        if (min_value > pearson_value) :
            min_value = pearson_value
            min_col_name = df.columns[n_col]
    return min_col_name,min_value

#피어슨 계수가 가장 높은 열의 이름과 값 반환
def cal_df_max_pearson_by(df, col_name) :
    max_value = -1;
    for n_col in range(0, len(df.columns)) :
        pearson_value = df[col_name].corr(df[df.columns[n_col]])
        if (max_value < pearson_value and pearson_value != 1.0) :
            max_value = pearson_value
            max_col_name = df.columns[n_col]
    return max_col_name,max_value

#데이터프레임 정규화
def norm_df(df,col_name) :
    x = df[col_name].values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x.astype(float))
    return pd.DataFrame(x_scaled, columns=col_name, index=df.index)

#검거율 합 정규화(?)(최대값을 100으로 변경하는 작업)
def norm_censorship_rate(df) :
    temp_max = df['검거율'].max()
    return df['검거율'] / temp_max * 100

#범죄발생수 정규화(?)(최대값을 1로 변경하는 작업)
def norm_crime_occurrence(df) :
    return df['범죄발생수'] / 5

#시각화 함수(너무 지저분해서 따로 만듦)
def make_pairplot(df) :
    sns.pairplot(df, x_vars=['CCTV현황', 'CCTV증가율'],
                 y_vars=['살인발생', '강도발생', '강제추행발생', '절도발생', '폭력발생'],
                 kind='reg')
    sns.pairplot(df, x_vars=['CCTV현황', 'CCTV증가율'],
                 y_vars=['살인검거율', '강도검거율', '강제추행검거율', '절도검거율', '폭력검거율'],
                 kind='reg')

#시각화 함수(너무 지저분해서 따로 만듦)
def make_heatmap(df,sort_by_col_name,heat_map_by_col_names,title_name) :
    df_sort = df.sort_values(by = sort_by_col_name, ascending = False)
    plt.figure(figsize=(10, 10))
    sns.heatmap(df_sort[heat_map_by_col_names], annot=True, fmt='f', linewidths=.5, cmap='RdPu')
    plt.title(title_name)

#파일 경로 설정
input_file = '..\Data'
font_path =  'C:/Windows/Fonts/malgun.ttf'

#시각화용 폰트
if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    font_name = font_manager.FontProperties(fname = font_path).get_name()
    rc('font', family=font_name)
else:
    print('폰트에러><')

#데이터 파일 경로 저장
allFile_list = glob.glob(os.path.join(input_file, '*'))

#csv저장 변수
allData = []

# [0] : crime_2014 [1] : crime_2015 [2] : crime_2016 [3] : seoul_cctv
#csv 저장
for file in allFile_list :
    df = pd.read_csv(file, encoding = 'euc-kr')
    allData.append(df)

#병합용 데이터 프레임
main_df = pd.DataFrame()

# print(allFile_list[3])

#데이터 프레임 열 이름 및 인덱스 정리
for n in range(0,len(allData)) :
    if(allFile_list[n] == '..\Data\seoul_cctv.csv') :
        #열이름 변경
        allData[n].columns = ['자치구','합계','2013년','2014년','2015년','2016년']
        #필요없는 열 삭제
        del allData[n]['합계']
        #CCTV증가율 계산
        #CCTV누적량 계산(기존 값은 해당 년도에 증가한 CCTV의 수)
        allData[n]['2014년증가율'] = allData[n]['2014년'] / allData[n]['2013년'] * 100
        allData[n]['2014년'] = allData[n]['2013년'] + allData[n]['2014년']
        allData[n]['2015년증가율'] = allData[n]['2015년'] / allData[n]['2014년'] * 100
        allData[n]['2015년'] = allData[n]['2014년'] + allData[n]['2015년']
        allData[n]['2016년증가율'] = allData[n]['2016년'] / allData[n]['2015년'] * 100
        allData[n]['2016년'] = allData[n]['2015년'] + allData[n]['2016년']
    else :
        #열이름 변경
        allData[n].columns = ['년도','자치구','총범죄발생','총범죄검거','살인발생','살인검거','강도발생','강도검거',
                              '강제추행발생','강제추행검거','절도발생','절도검거','폭력발생','폭력검거']
        # 의미 없는 행 삭제 인덱스 재정리
        allData[n] = allData[n].drop([0,1])
        allData[n] = allData[n].reset_index(drop = True)
        #main_df에 데이터 추가
        main_df = main_df.append(allData[n])

#CCTV현황 열 추가
main_df['CCTV현황'] = np.nan
#CCTV증가율 열 추가
main_df['CCTV증가율'] = np.nan
#열 순서 정렬
main_df = pd.DataFrame(main_df,columns = ['자치구','년도','CCTV현황','총범죄발생','총범죄검거','살인발생','살인검거',
                                          '강도발생','강도검거','강제추행발생','강제추행검거','절도발생','절도검거',
                                          '폭력발생','폭력검거','CCTV증가율'])
#데이터 자치구, 년도 별 정렬
main_df = main_df.sort_values(by=['자치구', '년도'])
#인덱스 재정렬
main_df = main_df.reset_index(drop = True)

#main_df에 CCTV수 추가
for r_main in range(0, len(main_df.axes[0])) :
    for r_cctv in range(0, len(allData[3].axes[0])) :
        if(main_df.iloc[r_main]['자치구'] == allData[3].iloc[r_cctv]['자치구']) :
            if(main_df.iloc[r_main]['년도'] == '2014') :
                main_df.set_value(r_main,'CCTV현황',allData[3].iloc[r_cctv]['2014년'])
                main_df.set_value(r_main, 'CCTV증가율', allData[3].iloc[r_cctv]['2014년증가율'])
            elif(main_df.iloc[r_main]['년도'] == '2015') :
                main_df.set_value(r_main, 'CCTV현황', allData[3].iloc[r_cctv]['2015년'])
                main_df.set_value(r_main, 'CCTV증가율', allData[3].iloc[r_cctv]['2015년증가율'])
            elif (main_df.iloc[r_main]['년도'] == '2016') :
                main_df.set_value(r_main, 'CCTV현황', allData[3].iloc[r_cctv]['2016년'])
                main_df.set_value(r_main, 'CCTV증가율', allData[3].iloc[r_cctv]['2016년증가율'])

#튜플들 쉼표 제거
main_df.replace(regex=True,inplace=True,to_replace=r'\W',value=r'')

#숫자형으로 형변환
astype_df_columns_to_numeric(main_df, ['년도','CCTV현황','총범죄발생','총범죄검거','살인발생','살인검거','강도발생',
                                       '강도검거','강제추행발생','강제추행검거','절도발생','절도검거','폭력발생',
                                       '폭력검거'])

#검거율 열 추가
main_df['살인검거율'] = main_df['살인검거']/main_df['살인발생']*100
main_df['강도검거율'] = main_df['강도검거']/main_df['강도발생']*100
main_df['강제추행검거율'] = main_df['강제추행검거']/main_df['강제추행발생']*100
main_df['절도검거율'] = main_df['절도검거']/main_df['절도발생']*100
main_df['폭력검거율'] = main_df['폭력검거']/main_df['폭력발생']*100

#검거율이 100을 넘어가는 경우 100으로 변경(발생건수보다 검거건수가 더 많은 경우가 많이 존재)
main_df.loc[main_df['살인검거율'] > 100, '살인검거율'] = 100
main_df.loc[main_df['강도검거율'] > 100, '강도검거율'] = 100
main_df.loc[main_df['강제추행검거율'] > 100, '강제추행검거율'] = 100
main_df.loc[main_df['절도검거율'] > 100, '절도검거율'] = 100
main_df.loc[main_df['폭력검거율'] > 100, '폭력검거율'] = 100

#피봇 설정
df_set_pivot = main_df.pivot_table(main_df, index=['자치구','년도'])

#CCTV현황과 증가율 데이터 분리하기
#cctv는 cctv현황, icctv는 cctv증가율
df_by_cctv = df_set_pivot[['CCTV현황','총범죄발생','총범죄검거','살인발생','살인검거','강도발생','강도검거',
                                          '강제추행발생','강제추행검거','절도발생','절도검거','폭력발생','폭력검거']]
df_by_icctv =  df_set_pivot[['CCTV증가율','총범죄발생','총범죄검거','살인발생','살인검거','강도발생','강도검거',
                                          '강제추행발생','강제추행검거','절도발생','절도검거','폭력발생','폭력검거']]
#검거율 데이터 프레임 생성
df_censorship_cctv = df_set_pivot[['CCTV현황','살인검거율','강도검거율','강제추행검거율','절도검거율','폭력검거율']]
df_censorship_icctv = df_set_pivot[['CCTV증가율','살인검거율','강도검거율','강제추행검거율','절도검거율','폭력검거율']]

#년도별로 데이터 프레임 생성 및 인덱스 재정렬 및 피봇 설정
df_2014 = main_df[main_df['년도'] == 2014];df_2014 = df_2014.reset_index(drop = True);df_2014 = df_2014.pivot_table(df_2014, index=['자치구','년도'])
df_2015 = main_df[main_df['년도'] == 2015];df_2015 = df_2015.reset_index(drop = True);df_2015 = df_2015.pivot_table(df_2015, index=['자치구','년도'])
df_2016 = main_df[main_df['년도'] == 2016];df_2016 = df_2016.reset_index(drop = True);df_2016 = df_2016.pivot_table(df_2016, index=['자치구','년도'])

#모든 데이터에서 피어슨 낮은 열의 이름 구하는 부분
pearson_min_col_name = cal_df_min_pearson_by(df_by_cctv,'CCTV현황')
pearson_min_col_name2 = cal_df_min_pearson_by(df_by_icctv,'CCTV증가율')
#검거율에서 피어슨 높은 열의 이름 구하는 부분
pearson_max_col_name = cal_df_max_pearson_by(df_censorship_cctv,'CCTV현황')
pearson_max_col_name2 = cal_df_max_pearson_by(df_censorship_icctv,'CCTV증가율')

#피어슨 값과 열 출력
print(pearson_min_col_name,pearson_min_col_name2,pearson_max_col_name,pearson_max_col_name2)

#열 이름들
cctv_col = ['CCTV현황','CCTV증가율']
crime_col = ['살인발생', '강도발생', '강제추행발생', '절도발생', '폭력발생','살인검거율','강도검거율','강제추행검거율',
             '절도검거율','폭력검거율']
crime_occurrence = ['살인발생', '강도발생', '강제추행발생', '절도발생', '폭력발생']
crime_censorship_rate = ['살인검거율','강도검거율','강제추행검거율','절도검거율','폭력검거율']

#정규화(범죄 발생수에 대한)
df_norm = norm_df(df_by_cctv,crime_occurrence)
df_norm_2014 = norm_df(df_2014,crime_occurrence)
df_norm_2015 = norm_df(df_2015,crime_occurrence)
df_norm_2016 = norm_df(df_2016,crime_occurrence)

#정규화 데이터 프레임에 검거율 추가
df_norm[crime_censorship_rate] = df_censorship_cctv[crime_censorship_rate]
df_norm_2014[crime_censorship_rate] = df_2014[crime_censorship_rate]
df_norm_2015[crime_censorship_rate] = df_2015[crime_censorship_rate]
df_norm_2016[crime_censorship_rate] = df_2016[crime_censorship_rate]

#범죄발생수 열 추가
df_norm['범죄발생수'] = np.sum(df_norm[crime_occurrence], axis = 1)
df_norm_2014['범죄발생수'] = np.sum(df_norm_2014[crime_occurrence], axis = 1)
df_norm_2015['범죄발생수'] = np.sum(df_norm_2015[crime_occurrence], axis = 1)
df_norm_2016['범죄발생수'] = np.sum(df_norm_2016[crime_occurrence], axis = 1)

#범죄발생수 정규화(?) 부분(최대값을 1로 변경하는 작업)
df_norm['범죄발생수'] = norm_crime_occurrence(df_norm)
df_norm_2014['범죄발생수'] = norm_crime_occurrence(df_norm_2014)
df_norm_2015['범죄발생수'] = norm_crime_occurrence(df_norm_2015)
df_norm_2016['범죄발생수'] = norm_crime_occurrence(df_norm_2016)

#검거율 열 추가
df_norm['검거율'] = np.sum(df_norm[crime_censorship_rate], axis = 1)
df_norm_2014['검거율'] = np.sum(df_norm_2014[crime_censorship_rate], axis = 1)
df_norm_2015['검거율'] = np.sum(df_norm_2015[crime_censorship_rate], axis = 1)
df_norm_2016['검거율'] = np.sum(df_norm_2016[crime_censorship_rate], axis = 1)

#검거율 합 정규화(?) 부분(최대값을 100으로 변경하는 작업)
df_norm['검거율'] = norm_censorship_rate(df_norm)
df_norm_2014['검거율'] = norm_censorship_rate(df_norm_2014)
df_norm_2015['검거율'] = norm_censorship_rate(df_norm_2015)
df_norm_2016['검거율'] = norm_censorship_rate(df_norm_2016)

#CCTV관련 열 추가
df_norm[cctv_col] = df_set_pivot[cctv_col]
df_norm_2014[cctv_col] = df_2014[cctv_col]
df_norm_2015[cctv_col] = df_2015[cctv_col]
df_norm_2016[cctv_col] = df_2016[cctv_col]

#시각화 부분
#주석 해제해서 하나씩 출력

# make_pairplot(df_norm)
# make_pairplot(df_norm_2014)
# make_pairplot(df_norm_2015)
# make_pairplot(df_norm_2016)

# make_heatmap(df_norm,'범죄발생수',crime_occurrence,'범죄비율 (정규화된 발생 건수로 정렬)')
# make_heatmap(df_norm,'검거율',crime_censorship_rate,'범죄 검거 비율 (정규화된 검거의 합으로 정렬)')

# make_heatmap(df_norm_2014,'범죄발생수',crime_occurrence,'범죄비율 (정규화된 발생 건수로 정렬)')
# make_heatmap(df_norm_2014,'검거율',crime_censorship_rate,'범죄 검거 비율 (정규화된 검거의 합으로 정렬)')

# make_heatmap(df_norm_2015,'범죄발생수',crime_occurrence,'범죄비율 (정규화된 발생 건수로 정렬)')
# make_heatmap(df_norm_2015,'검거율',crime_censorship_rate,'범죄 검거 비율 (정규화된 검거의 합으로 정렬)')

# make_heatmap(df_norm_2016,'범죄발생수',crime_occurrence,'범죄비율 (정규화된 발생 건수로 정렬)')
# make_heatmap(df_norm_2016,'검거율',crime_censorship_rate,'범죄 검거 비율 (정규화된 검거의 합으로 정렬)')

#시각화 출력부분
plt.show()

#각 데이터 프레임 csv로 저장
main_df.to_csv('../Result/main_df.csv', sep=',',encoding='euc-kr')
df_set_pivot.to_csv('../Result/df_set_pivot.csv', sep=',',encoding='euc-kr')
df_by_cctv.to_csv('../Result/df_by_cctv.csv', sep=',',encoding='euc-kr')
df_by_icctv.to_csv('../Result/df_by_icctv.csv', sep=',',encoding='euc-kr')
df_censorship_cctv.to_csv('../Result/df_censorship_cctv.csv', sep=',',encoding='euc-kr')
df_censorship_icctv.to_csv('../Result/df_censorship_icctv.csv', sep=',',encoding='euc-kr')
df_2014.to_csv('../Result/df_2014.csv', sep=',',encoding='euc-kr')
df_2015.to_csv('../Result/df_2015.csv', sep=',',encoding='euc-kr')
df_2016.to_csv('../Result/df_2016.csv', sep=',',encoding='euc-kr')
df_norm.to_csv('../Result/df_norm.csv', sep=',',encoding='euc-kr')
df_norm_2014.to_csv('../Result/df_norm_2014.csv', sep=',',encoding='euc-kr')
df_norm_2015.to_csv('../Result/df_norm_2015.csv', sep=',',encoding='euc-kr')
df_norm_2016.to_csv('../Result/df_norm_2016.csv', sep=',',encoding='euc-kr')