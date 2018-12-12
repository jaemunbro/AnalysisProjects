library(rvest)
library(stringr)
library(plyr)
library(dplyr)
library(ggvis)
library(knitr)


library(lubridate)
library(quantmod)

library(httr)
library(xml2)
library(readr)
library(rvest)

library(data.table)


?Sys.getlocale
###종목정보 가져올 때 한글인코딩이 안된 채로 나온다.. 한글 인코딩 어떻게 할지? encoding = "UTF-8"
Sys.getlocale()
#한국어 로케일설정시도
options()
Sys.setlocale("LC_ALL") 
# Sys.getlocale()
unlist(strsplit(Sys.getlocale(), ";")) 
# English_United States.1252  -.> Korean_Korea.949로 바뀌어야
Sys.setlocale("LC_COLLATE", "ko_KR.UTF-8")


#기본주소 + 페이지 넘버 합치기
url <- "http://consensus.hankyung.com/hankyung/change/updowns/index.php?func=browse&mode=&showtype=tprice_up&page="
url_pages <- paste(url, "240", sep = "")

? content
? read_html
#install.packages("httr")
library(httr)
library(xml2)
x <- content(GET(url), "raw")

#페이지별로 html못 읽는 이유 encoding 때문이었음..
#encoding 방식 확인
#install.packages("readr")
library(readr)

?? guess_encoding
guess_encoding(x)

#페이지별html
report_html <- read_html(url_pages, encoding = "EUC-KR")


# #ID .CLASS NAME
report_scrp <- report_html %>%
  html_nodes(".btext tr td") %>%
  html_text()


#data frame에 넣기
#df_report_scrp <- data.frame(report_scrp, stringsAsFactors = FALSE) 
#일단 array에 넣고 dataframe으로 넘기고....
array_report_scrp <- array(report_scrp, dim = c(9,16)) # 벡터를 9*16 배열로 배치
#어떻게 회전? 한row씩 조회해서 회전?
tmp_report_scrp <- data.frame(array_report_scrp, stringsAsFactors = F)
#FOR문으로 데이터 한개식 쌓기?  RESHAPE?
#FOR문 #http://rfriend.tistory.com/238 for문으로 다시 쌓기 
#tmp2_report_scrp <- tmp_report_scrp[2,2]
#RESHAPE써보기 #출처: http://rfriend.tistory.com/80 [R, Python 분석과 프로그래밍 (by R Friend)]
#install.packages("reshape")
#library(reshape)
#?melt
#Cars93_sample_melt <- melt(data = tmp_report_scrp, 
#                           id.vars = c(tmp_report_scrp[2,], tmp_report_scrp[3,]))

########################################
# rotation T # https://stackoverflow.com/questions/28680994/converting-rows-into-columns-and-columns-into-rows-using-r                      
df_report_scrp <- as.data.frame(t(tmp_report_scrp), stringsAsFactors = FALSE)
# header setting
colnames(df_report_scrp) = df_report_scrp[1, ] # the first row will be the header
df_report_scrp = df_report_scrp[-1, ]          # removing the first row.
df_report_scrp



#for문내 bind할 dataframe 먼저 생성
all_reports <- data.frame(matrix(nrow=0, ncol=9), stringsAsFactors = F)

####################################
# 2016년 report 전체 가져오기 for문 
#x : page numbers on hk consensus
#6/24 기준PAGE # 102~249 2220개 
####################################
for ( x in 102:249) {
  # 기본주소 + 페이지 넘버 합치기
  url <- "http://consensus.hankyung.com/hankyung/change/updowns/index.php?func=browse&mode=&showtype=tprice_up&page="
  url_pages <- paste(url, x, sep = "")
  
  #페이지별html
  report_html <- read_html(url_pages, encoding = "EUC-KR")
  
  #스크랩핑 (#ID .CLASS NAME)
  report_scrp <- report_html %>%
    html_nodes(".btext tr td") %>%
    html_text()
  
  #일단 array에 넣고 dataframe으로 넘기고....
  array_report_scrp <- array(report_scrp, dim = c(9,16)) # 벡터를 9*16 배열로 배치
  tmp_report_scrp <- data.frame(array_report_scrp, stringsAsFactors = FALSE)
  
  # rotation T # https://stackoverflow.com/questions/28680994/converting-rows-into-columns-and-columns-into-rows-using-r                      
  df_report_scrp <- as.data.frame(t(tmp_report_scrp), stringsAsFactors = FALSE)
  
  # header setting
  #colnames(df_report_scrp) = df_report_scrp[1, ] # the first row will be the header
  df_report_scrp = df_report_scrp[-1, ]          # removing the first row.
  df_report_scrp = df_report_scrp[, -1]          # removing the first col.
  
  # binding data
  all_reports = rbind(all_reports, df_report_scrp)
}

#name of colurms setting
#names(all_reports) <- c("stk_nm", "의견날짜", "목표가", "이전목표가", "투자의견일종가", "투자의견", "기존투자의견", "증권사")
names(all_reports) <- c("stk_nm", "의견날짜", "목표가", "이전목표가", "투자의견일종가", "투자의견", "기존투자의견", "증권사")

all_reports
str(all_reports)

#종목명 trim 함수 만들기
trim <- function (x) gsub("^\\s+|\\s+$", "", x)
trim(" aa bbb    ")

all_reports$stk_nm <- trim(all_reports$stk_nm)

#######################
### 주식 가격 받아오기 
#######################
#install.packages("quantmod")
library(quantmod)

#KS 없이도 표출 가능 
tail(getSymbols("005930", env=NULL, src = "google"))[,4]

#주식 종목코드와 종목명 매칭
#종목코드 종목명 제공 사이트정보 이용 
stock_id_nm_html <- read_html("http://bigdata-trader.com/itemcodehelp.jsp")


#종목코드 종목명 추출 
stk_id <- stock_id_nm_html %>%
  html_nodes("tr td:nth-child(1)") %>%
  html_text()

stk_nm <- stock_id_nm_html %>%
  html_nodes("tr td:nth-child(2)") %>%
  html_text()

stk_grp <- stock_id_nm_html %>%
  html_nodes("tr td:nth-child(3)") %>%
  html_text()


# 종목명 종목id 종목그룹 data frame
id_nm_matching <- data.frame(trim(stk_id),trim(stk_nm),trim(stk_grp),stringsAsFactors = FALSE)
#names setting
tmp1 <- c("stk_id", "stk_nm", "stk_grp")
names(id_nm_matching) <- tmp1

all_reports$stk_nm
id_nm_matching$stk_nm

#all_reports에 있는 종목명과 id_nm_matching의 종목명과 JOIN -> 종목코드 추출 (JOIN)
stk_merge <- merge(x = id_nm_matching,
                   y = all_reports,
                   by = "stk_nm")

###################################################
# calculating dates
###################################################
#install.packages("lubridate")
library(lubridate)

#make date data tidy 
gsub("/", "", stk_merge$의견날짜)

#change data format
ymd(gsub("/", "", stk_merge$의견날짜))

#하루 뒤 test
as.Date(ymd(gsub("/", "", stk_merge$의견날짜))) + days(1)
#6개월 뒤 test
as.Date(ymd(gsub("/", "", stk_merge$의견날짜))) %m+% months(6)

#stk_kospi$의견날짜 data 변경
stk_merge$의견날짜 <- ymd(gsub("/", "", stk_merge$의견날짜))

###################################################
# Separating KOSPI KOSDAQ
###################################################
stk_kospi <- stk_merge[stk_merge$stk_grp == "KOSPI",]
stk_kosdaq <- stk_merge[stk_merge$stk_grp == "KOSDAQ",]

#KOSPI KOSDAQ별 리포트 갯수
nrow(stk_kospi) #1783
nrow(stk_kosdaq) #407

#stk_kospi 주가 정보 test
getSymbols("KOSPI:033780",  env=NULL, src = "google"
           ,from = "2017-01-02"
           ,to   = "2017-01-05")[,4]

?getSymbols.yahoo

## df setting for for문
stk_prc_tracking <- data.frame(matrix(nrow=0, ncol=11), stringsAsFactors = F)

###warning : getSymbols실행 시 실패하는 종목코드들 삭제 000120, 002550 002380 051900???
#너무많아서 다시 확인.. 앞에 KOSPI: 붙여주면 조회 된다.
stk_kospi <-subset(stk_kospi, stk_id != "002380")

stk_kospi$stk_id = "033780"

ncol(stk_kospi)

###################################################
# KOSPI 주가 tracking for문
# 1개월 뒤, 3개월 뒤, 6개월 뒤
#
# 기준일로부터 1개월뒤 일자가 주말인 경우 주가정보가 없으므로..
# 기준일로부터 1개월뒤 일자에 -2 +2일씩 더하여 5일간 주가를 일차로 가져온 후,그중 가장 높은 주가를 가져온다.
#
# 한번에 2200까지 for문돌리면 자꾸 에러가 남.. 10~100여개로 끊어서 시도하면 됐다가 안됐다가.. 이유는 추후에 찾아보자..
# 빠른 속도로 지속 접속 시도 시 google에서 접속을 차단하는 것으로 보임.
###################################################
for ( x in 201:300) {
  #for ( x in 1:nrow(stk_kospi)) {
  
  # if(nrow(stk_prc_tracking) > x ){ 
  #   x <-nrow(stk_prc_tracking) + 1
  # }
  
  tmp_df <- data.frame(stk_kospi[x,]
                       # #1개월뒤
                       # ,month_1_after=max(getSymbols(paste("KOSPI:",stk_kospi[x,2],sep=""),  env=NULL, src = "google"
                       #                               ,from = as.Date(stk_kospi$의견날짜[x] -2 ) %m+% months(1)
                       #                               ,to   = as.Date(stk_kospi$의견날짜[x] +2 ) %m+% months(1))[,4])
                       # #3개월뒤
                       # ,month_3_after=max(getSymbols(paste("KOSPI:",stk_kospi[x,2],sep=""),  env=NULL, src = "google"
                       #                               ,from = as.Date(stk_kospi$의견날짜[x] -2 ) %m+% months(3)
                       #                               ,to   = as.Date(stk_kospi$의견날짜[x] +2 ) %m+% months(3))[,4])
                       #6개월뒤
                       ,month_6_after=max(getSymbols(paste("KOSPI:",stk_kospi[x,2],sep=""),  env=NULL, src = "google"
                                                     ,from = as.Date(stk_kospi$의견날짜[x] -2 ) %m+% months(6)
                                                     ,to   = as.Date(stk_kospi$의견날짜[x] +2 ) %m+% months(6))[,4])
                       # #의견날짜KOSPI지수
                       # ,지수_의견일 =max(getSymbols("KOSPI",  env=NULL, src = "google"
                       #                         ,from = as.Date(stk_kospi$의견날짜[x] -2 ) 
                       #                         ,to   = as.Date(stk_kospi$의견날짜[x] +2 ))[,4])
                       # #6개월뒤KOSPI지수
                       # ,지수_6개월  =max(getSymbols("KOSPI",  env=NULL, src = "google"
                       #                          ,from = as.Date(stk_kospi$의견날짜[x] -2 ) %m+% months(6)
                       #                          ,to   = as.Date(stk_kospi$의견날짜[x] +2 ) %m+% months(6))[,4])
                       
  )
  
  ## row binding
  stk_prc_tracking <- rbind(stk_prc_tracking, tmp_df)
}

nrow(stk_prc_tracking)
stk_prc_tracking

str(stk_prc_tracking)
tail(stk_prc_tracking)



#######################################################
###     Getting historical stock price data from google finance
###     Coding by lisist, Jun 2015
#######################################################

### You need XML package to run this function
library(XML)
###########################################


g.hist <- function(ticker,start_date,end_date){
  
  # for debugging
  # g.hist("KOSDAQ:035720","2015-01-01","2015-01-15")
  #ticker<-"KOSDAQ:035720"
  #start_date<-"2015-01-01"
  #end_date<-"2017-04-16"
  
  #### Ticker modify ####
  ticker_m <- gsub(":","%3A",ticker)
  
  #### Date modify #### 
  start_date <- as.character(start_date)
  end_date <- as.character(end_date)
  start_date_m <- as.character(as.Date(start_date),format="%b:%d:%Y")
  end_date_m <- as.character(as.Date(end_date),format="%b:%d:%Y")
  start_date_m <- paste(strsplit(start_date,":")[[1]][1],'+',strsplit(start_date,":")[[1]][2],'%2C+',strsplit(start_date,":")[[1]][3],sep='')
  end_date_m <- paste(strsplit(end_date,":")[[1]][1],'+',strsplit(end_date,":")[[1]][2],'%2C+',strsplit(end_date,":")[[1]][3],sep='')
  
  
  #### Setting URL ####
  url ='http://www.google.com/finance/historical?q='
  url = paste(url,ticker_m,'&startdate=',start_date_m,'&enddate=',end_date_m,'&num=200',sep='')
  
  
  #### downloading data from the Google Finance ####
  output <- as.data.frame(readHTMLTable(url,stringsAsFactors=FALSE)[4])
  #head(output)
  #tail(output)
  #NROW(output)
  lct<-Sys.getlocale("LC_TIME")
  Sys.setlocale("LC_TIME", "C")
  output[,1] <- as.Date(output[,1],format="%b %d, %Y")
  Sys.setlocale("LC_TIME", lct)  
  names(output) <- c("Date",paste(ticker,c("Open","High","Low","Close","Volume"),sep="."))
  for (k in 2:6){
    output[,k] <- as.numeric(gsub(",","",output[,k]))
  }
  # NROW(output)
  
  #### Checking whether all data are retrieved or not ####
  end <- as.Date(output[length(output[,1]),1])
  ll <- 200-length(output[,1])
  if ((end-ll)<start_date) end <- start_date
  if (end > start_date) {
    output <- rbind(output,g.hist(ticker,start_date,as.character(end-1)))
  }
  
  # NROW(output)
  # output  
  return(output)
}

#g.hist test
g.hist("KOSDAQ:035720","2014-01-02","2015-07-11")

month_1_after=max(g.hist(paste("KOSDAQ:",stk_kosdaq[1,2],sep="")
                         ,as.Date(stk_kosdaq$의견날짜[1] -2 ) %m+% months(3)
                         ,as.Date(stk_kosdaq$의견날짜[1] +2 ) %m+% months(3))[,4])


## df setting for for문
kosdaq_prc_tracking <- data.frame(matrix(nrow=0, ncol=11), stringsAsFactors = F)
## df name setting
# names(kosdaq_prc_tracking) <- c("stk_id", "month_1_after", "month_3_after", "month_6_after")

###################################################
# KOSDAQ 주가 tracking for문
# 1개월 뒤, 3개월 뒤, 6개월 뒤
#
# 기준일로부터 1개월뒤 일자가 주말인 경우 주가정보가 없으므로..
# 기준일로부터 1개월뒤 일자에 -2 +2일씩 더하여 5일간 주가를 일차로 가져온 후,그중 가장 높은 주가를 가져온다.
###################################################
for ( x in 363:400) {
  # for ( x in 1:nrow(stk_kosdaq)) {
  tmp_df <- data.frame(stk_kosdaq[x,]
                       # ,month_1_after=max(g.hist(paste("KOSDAQ:",stk_kosdaq[x,2],sep="")
                       #                           ,as.Date(stk_kosdaq$의견날짜[1] -2 ) %m+% months(1)
                       #                           ,as.Date(stk_kosdaq$의견날짜[1] +2 ) %m+% months(1))[,4])
                       # ,month_3_after=max(g.hist(paste("KOSDAQ:",stk_kosdaq[x,2],sep="")
                       #                           ,as.Date(stk_kosdaq$의견날짜[1] -2 ) %m+% months(3)
                       #                           ,as.Date(stk_kosdaq$의견날짜[1] +2 ) %m+% months(3))[,4])
                       ,month_6_after=max(g.hist(paste("KOSDAQ:",stk_kosdaq[x,2],sep="")
                                                 ,as.Date(stk_kosdaq$의견날짜[1] -2 ) %m+% months(6)
                                                 ,as.Date(stk_kosdaq$의견날짜[1] +2 ) %m+% months(6))[,4])
                       # #의견날짜KOSDAQ지수
                       # ,지수_의견일 =max(g.hist("KOSDAQ:KOSDAQ"
                       #                     ,as.Date(stk_kosdaq$의견날짜[1] -2 )
                       #                     ,as.Date(stk_kosdaq$의견날짜[1] +2 ))[,4])
                       # #6개월뒤KOSPI지수
                       # ,지수_6개월 =max(g.hist("KOSDAQ:KOSDAQ"
                       #                     ,as.Date(stk_kosdaq$의견날짜[1] -2 ) %m+% months(6)
                       #                     ,as.Date(stk_kosdaq$의견날짜[1] +2 ) %m+% months(6))[,4])
  )
  
  ## row binding
  kosdaq_prc_tracking <- rbind(kosdaq_prc_tracking, tmp_df)
}

str(kosdaq_prc_tracking)
#############################
#최종! KOSPI KODAQ BINDING
#############################
stk_kospi_kodaq_prc <- rbind(stk_prc_tracking, kosdaq_prc_tracking)

#종목별 리포트 발행 갯수
num_of_reports <- unique(stk_kospi_kodaq_prc$stk_id) #175개
#중복 발행 갯수
num_of_reports_per_stk <- data.frame(table(stk_kospi_kodaq_prc$stk_id))

#############################
# 상승 비율 계산
# (기준시점 리포트 주가 - 발행당시 주가) / 발행당시 주가 * 100
#############################

#6개월뒤 상승률 
stk_kospi_kodaq_prc$prv_prs_ratio06 <- (stk_kospi_kodaq_prc$month_6_after - as.numeric(stk_kospi_kodaq_prc$투자의견일종가)) / as.numeric(stk_kospi_kodaq_prc$투자의견일종가) * 100
#상승률 Na 뺀다. 계산이 안됨.
stk_kospi_kodaq_prc <- subset(stk_kospi_kodaq_prc, stk_kospi_kodaq_prc$prv_prs_ratio06 != "NA") #NA제외
#BGF리테일, 에스에프에이 등 무상증자종목 때문에 결과 왜곡.. 어떻게 할까?
#일단50미만 종목만 뽑음 stk_kospi_kodaq_prc2
stk_kospi_kodaq_prc2 <- subset(stk_kospi_kodaq_prc, stk_kospi_kodaq_prc$prv_prs_ratio06 < -50) #-50미만 종목만 뽑는다.

str(stk_kospi_kodaq_prc2)
unique(stk_kospi_kodaq_prc2$stk_id)[2]

# 해당 id로만 따로 df 생성 
stk_kospi_kodaq_prc4 <- data.frame(matrix(nrow=0, ncol=12), stringsAsFactors = F)

stk_kospi_kodaq_prc2[stk_kospi_kodaq_prc2$stk_id == unique(stk_kospi_kodaq_prc2$stk_id)[2],]

#"027410" "051500" "222040" "090460" "056190" "138690" "030530" "200130"
for(x in 1:8){
  stk_kospi_kodaq_prc3 <- stk_kospi_kodaq_prc2[stk_kospi_kodaq_prc2$stk_id == unique(stk_kospi_kodaq_prc2$stk_id)[x],]
  stk_kospi_kodaq_prc4 <- rbind(stk_kospi_kodaq_prc4, stk_kospi_kodaq_prc3)
  }

stk_kospi_kodaq_prc4

# 기존 종목목록에서는 지운다
#install.packages("data.table")
library(data.table)
dt1 <- data.table(stk_kospi_kodaq_prc, key="stk_id")
dt2 <- data.table(stk_증자)
stk_kospi_kodaq_prc3 <- dt1[!dt2]

stk_증자 <-stk_증자[,1:12]

# 증자 종목들은 1/2 하고
stk_증자$투자의견일종가.x <- as.numeric(stk_증자$투자의견일종가.x)/2 
stk_증자$목표가.x <- as.numeric(stk_증자$목표가.x)/2

str(stk_kospi_kodaq_prc3)
stk_kospi_kodaq_prc3 <- data.frame(stk_kospi_kodaq_prc3, stringsAsFactors = F)
names(stk_kospi_kodaq_prc3)
colnames(stk_증자) <- names(stk_kospi_kodaq_prc3) 
#증자 제외df 와 증자 df 두df를 rbind 한다
stk_kospi_kodaq_prc <- rbind(stk_kospi_kodaq_prc3, stk_증자)

#무상증자 종목 계산 후 상승률 다시 구한다.
stk_kospi_kodaq_prc$prv_prs_ratio06 <- (stk_kospi_kodaq_prc$month_6_after - as.numeric(stk_kospi_kodaq_prc$투자의견일종가)) / as.numeric(stk_kospi_kodaq_prc$투자의견일종가) * 100

#지수상승률  
#stk_kospi_kodaq_prc$index_ratio06 <- (stk_kospi_kodaq_prc$지수_6개월 - stk_kospi_kodaq_prc$지수_의견일) / stk_kospi_kodaq_prc$지수_의견일 * 100

#주가 상승률 - 지수 상승률  
# stk_kospi_kodaq_prc$index_ratio06 <- stk_kospi_kodaq_prc$prv_prs_ratio06 - stk_kospi_kodaq_prc$index_ratio06

#목표주가와 실제주가 사이 GAP %
# stk_kospi_kodaq_prc$tgt_prs_ratio06 <- (as.numeric(stk_kospi_kodaq_prc$목표가) - stk_kospi_kodaq_prc$month_6_after) / stk_kospi_kodaq_prc$month_6_after * 100

#6개월뒤 상승갯수
cnt_up <- nrow(subset(stk_kospi_kodaq_prc, prv_prs_ratio06 > 0)) #  448

#3월 상승갯수
# cnt_up_03 <- nrow(subset(stk_kospi_kodaq_prc, prv_prs_ratio03 > 0)) #

#하락갯수
cnt_down <- nrow(subset(stk_kospi_kodaq_prc, prv_prs_ratio06 < 0)) # 752

subset(stk_kospi_kodaq_prc, prv_prs_ratio06 < -40)

str(stk_kospi_kodaq_prc)

#all stocks : AVG SUM / 970  = 평균상승
avg_up_down <- sum(stk_kospi_kodaq_prc$prv_prs_ratio06) / 970 #-1.7%

#KOSPI
avg_ksp_list <- subset(stk_kospi_kodaq_prc, stk_grp == "KOSPI")
avg_ksp_list$prv_prs_ratio06
str(avg_ksp_list) #579
#KOSDAQ 
avg_ksd_list <- subset(stk_kospi_kodaq_prc, stk_grp == "KOSDAQ")
avg_ksd_list$prv_prs_ratio06
str(avg_ksd_list) #391
#KOSPI avg 
avg_up_ksp <- sum(avg_ksp_list$prv_prs_ratio06) / 579 #-2.8%

avg_up_ksd <- sum(avg_ksd_list$prv_prs_ratio06) / 391 #-2.8%


#AVG SUM GAP / 60
avg_gap_05 <- sum(stk_kospi_kodaq_prc$tgt_prs_ratio05) /60

#차트 그리기
ggplot(diamonds, aes(depth, fill = cut, colour = cut)) +
  geom_density(alpha = 0.1) +
  xlim(55, 70)
