#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns #Visulization

sns.set()

from scipy.stats import norm, skew



#root_meam_squared_error
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


# In[2]:


def categ_or_contin(df, criterion=10, cat_or_cont=True, print_col = True):
    '''
    df의 column을 이루고 있는 value들 종류 갯수를 파악하여 연속형 변수와 범주형 변수로 구분해주는 함수.
    cat_or_cont = True (default) : 연속형 변수 출력
    cat_or_cont = False : 범주형 변수 출력
    criterion(default = 10) : 연속과 범주형의 기준 제시.
    ex) criterion = 10 : value의 종류가 10종이하이면 범주형 변수, 넘으면 연속형 변수
    ex) categ_or_contin(house6[0],15,False)
    '''   
    target_skew = df
    categorical = []
    continuous = []

    for column in target_skew: 
        if len(set(target_skew[column])) > criterion: # 범주형과 연속형 변수 구분 필요함. 
            continuous.append(column)  # 연속형
        else:
            categorical.append(column) #범주형
    if print_col == True:
        print('- categorical variables : ', categorical, '\n- continuous variables : ', continuous)
    return continuous if cat_or_cont==True else categorical


# In[4]:


def test_for_transf(df, col_str, transf_ls):
    '''
    Continous variable의 적절한 전처리를 위하여 '입력된 수식들로 변환된 column' vs 'frequency'의 그래프를 출력하는 함수.
    (사용시 추천예시) : 이렇게 같은 cell에 variable들을 정의하고, 함수안에선 수식만 바꾸면 더 편하다.
    df = house7[0]; col_str = 'bathrooms'; x = df[col_str];
    test_for_transf(df, col_str ,[x, np.log(x), (x-x.min())**(2/3)] )

    df : 해당 데이터프레임 입력
    col_str : df에서 확인해보고 싶은 column 하나를 string으로 입력 ex) 'price'
    transf_ls : 'del x'를 선행한 후 테스트 해보고 싶은 transfromation 수식을 target에대한 함수로 만들어서 list형태로 입력 
        ex) [(x-x.min())^2, np.log(x)]
    '''
    target_col = col_str
    target = df[target_col]
    target_transf = transf_ls # test해보고 싶은 transf 수식들 다 list 안에 넣기
    for i in range(len(target_transf)):
        # let's plot a histogram with the fitted parameters used by the function
        sns.distplot(target_transf[i] , fit=norm);
        # get mean and standard deviation
        (mu, sigma) = norm.fit(target_transf[i])
        # add legends to the plot
        plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                    loc='best')
        plt.ylabel('Frequency')
#        plt.title(target_col)
        plt.show()


# In[5]:


def check_skew(df, column_list_=None, skewness=1.5, criterion_of_cont_var=15, print_col_=True):
    '''
    df의 column인 column_list 중 왜도(skewness)가 1.5(default)보다 큰 column을 출력하는 함수.
    column_list(default=df.column, True = 연속형, False = 범주형, list = 그 list) 
            : df의 column들 중 왜도를 확인해볼 column list 입력. 
    skewness(default=1.5) : 기준 왜도값 입력.
    print_col_(default=True) : True이면 연속형 변수와 범주형 변수들을 리스트 형태로 보여줌
    '''
    target_skew = df
    if column_list_ == None: #구분할 행을 입력하지 않으면 df의 전체 행을 입력해준다.
        column_list=list(df.columns)
    elif column_list_ == True:
        column_list=categ_or_contin(target_skew,criterion_of_cont_var,True, print_col=print_col_)
    elif column_list_ == False:
        column_list=categ_or_contin(target_skew,criterion_of_cont_var,False, print_col=print_col_)
    else :
        column_list=column_list_
    # df.skew(): 열별 왜도
    biased_condition = abs(target_skew[column_list].skew()) > skewness # biased_condition : 1.5보다 크면 치우쳤다고 간주함
    # biased_variables: 왜도의 절대 값이 1.5보다 큰 변수
    biased_variables = target_skew[column_list].columns[biased_condition] 
    print('Columns {}에서 \n skewness가 {}보다 높은 columns은 {}이다.\n'.format(column_list, skewness, list(biased_variables)))
    


# In[6]:


def Remove_outliers(IQR_target_df, criterion_of_cont_var=15, criterion_Q1=0.25, criterion_Q3=0.75):
    '''
    IQR rule에 따라 연속형 변수의 이상치를 제거해주는 함수.
    IQR rule : Q3-1.5*IQR보다 크거나 Q1-1.5*IQR보다 작으면 이상치라고 판단. 삭제
    IQR_target_df : 타겟이 될 데이터 프레임
    criterion_of_cont_var (default=15) : 연속형 변수의 기준.
    ex) Remove_outliers(house4_1[0], 20)
    '''
    IQR_target = IQR_target_df #house4_1[0]
    continuous_IQR = categ_or_contin(IQR_target,criterion_of_cont_var,print_col=False)

    len_before = len(IQR_target)
    Q1 = IQR_target[continuous_IQR].quantile(criterion_Q1)
    Q3 = IQR_target[continuous_IQR].quantile(criterion_Q3)
    IQR = Q3 - Q1
    condition1 = Q1 - 1.5*IQR < IQR_target[continuous_IQR]
    condition2 = Q3 + 1.5*IQR > IQR_target[continuous_IQR]
    IQR_target[continuous_IQR] = IQR_target[continuous_IQR][condition1 & condition2]
    # IQR rule : Q3-1.5*IQR보다 크거나 Q1-1.5*IQR보다 작으면 이상치라고 판단. 삭제
    IQR_target.dropna(inplace=True)
    IQR_target.reset_index(drop=True, inplace=True)
    len_after = len(IQR_target)
    print('Outliers are completely removed. Length is redeced from {} to {}'.format(len_before, len_after))


# In[7]:


def count_category(target, criterion_of_cont_var=15, sort_category = False):
    '''
    target의 columns 중 categorical variable의 category 종류 및 각 category에 대한 샘플의 수를 count해주는 함수.
    target : 원하는 dataframe을 입력.
    criterion_of_cont_var (default=10) : categorical variable의 기준을 입력.
        ex) criterion_of_cont_var=10 : variable이 10 종 이하로 전부 분류되면 categorical variable.
    sort_category (default = False) : False면 sample수가 많은 카테고리 순으로 배열, True면 카테고리를 오름차순으로 배열
    '''
    count_target = target
    result = []
    categ_var = categ_or_contin(count_target,criterion_of_cont_var,False,False)
    
    for i in range(len(categ_var)):
        pd_count = count_target[categ_var[i]].value_counts()
        pd_count = pd_count.reset_index().rename(columns = {'index': 'category'})
        if sort_category == True:
            pd_count.sort_values(by = ['category'], ascending=sort_category, inplace=True)
            pd_count.reset_index(drop=True, inplace=True)
        result.append(pd_count)
    return pd.concat(result,axis=1)


# In[8]:

def rowXcol_for_subfig(length):
    '''
    입력된 길이를 포함 할 수 있는 적절한 행과 열을 결정해준다. cf) row_len, col_len = rowXcol_for_subfig(len(df))
    ex) rowXcol_for_subfig(len(df)) : df의 feature(의 그래프)를 행렬로 만들기 위한 행의 길이와 열의 길이을 결정해준다.
    '''
    row_len = int(round(length**(1/2),0))
    col_len = int(round(length/row_len,0))+1 if row_len*round(length/row_len,0) < length else int(round(length/row_len,0))
    return row_len, col_len


def features_vs_frequency(df_target_, figsize_tuple_ = (12,10), loc_='best'):
    '''
    df_target_의 features_들의 분포를 행렬 그래프로 그리는 함수
    df_target_ : 그래프들 그려볼 data frame
    figsize_tuple_ : 전체 그래프 사이즈를 튜플로 입력
    loc_ : lengend 위치 설정
      ex) loc_ = 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 
                'center left', 'center right', 'lower center', 'upper center', 'center' 
    '''    
    df_target = df_target_
    figsize_tuple = figsize_tuple_
    
    col = df_target.columns # df_target의 feature list
    
    # data frame의 feature의 갯수에 따라 적절한 행과 열을 결정해준다. 
    row_len, col_len = rowXcol_for_subfig(len(col))
    print('그래프가 {}X{}행렬로 그려집니다.'.format(row_len, col_len))
    
    # let's plot a histogram with the fitted parameters used by the function
    fig = plt.figure(figsize=figsize_tuple)
    for i in range(len(col)):
        target = df_target[col[i]]
        axi = fig.add_subplot(row_len,col_len,i+1)
        sns.distplot(target , fit=norm, ax=axi)
        plt.ylabel('Frequency')
        plt.legend([col[i]], loc=loc_)


# In[9]:


def features_vs_label(df_target_, label_str , col_target=None, figsize_tuple_ = (12,10), loc_='best'):
    '''
    df_target_의 col_target vs label를 행렬 그래프로 그리는 함수
    df_target_ : 그래프들 그려볼 data frame
    label_str : df_target_의 label을 str 형태로 입력.
    col_target : label에 대한 분포를 파악하기를 원하는 features를 list형태로 입력.
    figsize_tuple_ : 전체 그래프 사이즈를 튜플로 입력
    loc_ : lengend 위치 설정
      ex) loc_ = 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 
                'center left', 'center right', 'lower center', 'upper center', 'center' 
    '''
    if col_target == None: #구분할 행을 입력하지 않으면 df의 전체 행을 입력해준다.
        col = list(df_target_.columns)
    else :
        col = col_target    
    
    df_target = df_target_
    figsize_tuple = figsize_tuple_
    
    
    # data frame의 feature의 갯수에 따라 적절한 행과 열을 결정해준다.
    row_len, col_len = rowXcol_for_subfig(len(col))
    print('그래프가 {}X{}행렬로 그려집니다.'.format(row_len, col_len))

    # columns vs label
    fig = plt.figure(figsize=figsize_tuple)
    for i in range(len(col)):
        axi = fig.add_subplot(row_len,col_len,i+1)
        sns.regplot(x=col[i], y=label_str, data=df_target, ax=axi)
        plt.legend([col[i]], loc=loc_)


# In[10]:


def one_feature_vs_freqency(df_target, col_name, loc_='best'):
    '''
    f_target의 feature인 col_name의 분포(frequency)를 행렬 그래프로 그리는 함수
    df_target : 그래프를 그려볼 data frame
    col_name : 그래프를 그려볼 featreu 이름 type = str
    loc_ : lengend 위치 설정
        ex) loc_ = 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 
            'center left', 'center right', 'lower center', 'upper center', 'center' 
    '''
    target = df_target[col_name]
    col_target = col_name
    
    # skewness 체크
    print("Skewness: %f" % target.skew())
    
    # let's plot a histogram with the fitted parameters used by the function
    sns.distplot(target , fit=norm);
    # get mean and standard deviation
    (mu, sigma) = norm.fit(target)
    # add legends to the plot
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                loc=loc_)
    plt.ylabel('Frequency')
    plt.title(col_target);


# In[11]:


def corr_heatmap(df_corr, sort_by_col, column_list_=None, figsize_tuple=(12, 10), ascending_=False, palette_color='purple', fmt_=".2f"):
    '''
    df_corr의 feature+label 사이의 correlation coefficient를 heatmap으로 나타내는 함수.
    df_corr : 상관계수 heatmap을 그려 볼 data frame
    sort_by_col : 상관계수를 sort_by_col 기준으로 정렬 cf) 보통 label로 설정
    fig_size (default=(12, 10)) : heatmap figure size
    ascending_ (default=False) : 상관계수 내림차순 정렬(False), 오름차순 정렬(True)
    palette_color (default='purple') : 상관계수 나타낼 색
    fmt_ (default=".2f") : 표시되는 상관계수의 자릿수. cf) ".2f": 소숫점 2자리수 float
    '''
    
    if column_list_ == None: #구분할 행을 입력하지 않으면 df의 전체 행을 입력해준다.
        column_list = list(df_corr.columns)
    else :
        column_list = column_list_
    
    # 상관계수 구해서 그래프 그리기.
    h_corr = df_corr[column_list].corr(method='pearson').sort_values(by=[sort_by_col], ascending = ascending_)                .sort_values(by=[sort_by_col], axis =1, ascending=ascending_)  
    # 상관계수 dataframe /  가격과 상관계수 큰 순으로 정렬(행과 열 모두다 정렬)
    
    #heatmap으로 그리기
    plt.figure(figsize=figsize_tuple)
    sns.heatmap(h_corr, cmap=sns.light_palette(palette_color, as_cmap=True), annot=True, fmt=fmt_)
    plt.show()







