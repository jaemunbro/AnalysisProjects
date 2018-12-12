######################################
# 1. 공공 API DATA 입수
######################################
### Using Open API
install.packages("httr")
library(httr)
library(dplyr)
library(xml2)

# 월세가격지수 조회 서비스
# http://data.insight.go.kr/openapi/service/PriceItemList/getPriceItemList?ServiceKey=LiPnwV3uVNCQj2gekrUZc8bEjNmk3HviTRN4KqwBm4sGSxbkZtfU6QtVzvJKwWYpbgq%2Fump6ndupnscz5WuHgw%3D%3D
# LiPnwV3uVNCQj2gekrUZc8bEjNmk3HviTRN4KqwBm4sGSxbkZtfU6QtVzvJKwWYpbgq%2Fump6ndupnscz5WuHgw%3D%3D

#http://openapi.kab.co.kr/OpenAPI_ToolInstallPackage/service/rest/RentPrcIndexSvc?_wadl&type=xml
rentprc <- GET("http://openapi.kab.co.kr/OpenAPI_ToolInstallPackage/service/rest/RentPrcIndexSvc/getRentPrcIndex?startmonth=201501&endmonth=201512&region=11000&serviceKey=LiPnwV3uVNCQj2gekrUZc8bEjNmk3HviTRN4KqwBm4sGSxbkZtfU6QtVzvJKwWYpbgq%2Fump6ndupnscz5WuHgw%3D%3D")

status_code(rentprc)

headers(rentprc)

content(rentprc, "raw")
content(rentprc, "text")

#바로 read_xml로 넣자.
##### XML Scraping
y<- read_xml("http://openapi.kab.co.kr/OpenAPI_ToolInstallPackage/service/rest/RentPrcIndexSvc/getRentPrcIndex?startmonth=201501&endmonth=201512&region=11000&serviceKey=LiPnwV3uVNCQj2gekrUZc8bEjNmk3HviTRN4KqwBm4sGSxbkZtfU6QtVzvJKwWYpbgq%2Fump6ndupnscz5WuHgw%3D%3D")

y<- read_xml(rentprc)
xml_name(y)
xml_children(y)
regionCd <- xml_find_all(y, ".//regionCd")
regionNm <- xml_find_all(y, ".//regionNm")
rsRow <- xml_find_all(y, ".//rsRow")
xml_path(regionCd)
regionCd <- xml_text(regionCd)
regionNm <- xml_text(regionNm)
rsRow <- xml_text(rsRow)
test <- unlist(strsplit(rsRow, "[|]"))

rent_price <- data_frame(regionCd = regionCd, regionNm = regionNm, rsRow = test)



#########################
# 2.MONGODB 연동
#########################
install.packages("mongolite")
library(mongolite)


con <- mongolite::mongo(collection = "skku.keywordset",
                        db = "admin",
                        url = "mongodb://localhost",
                        verbose = TRUE,
                        options = ssl_options())

#skku.keywordset collection Export 후 localhost mongodb에 import시킨 후 작업
m<-mongo(collection="skku.keywordset",db="admin", verbose=FALSE)

#17년도 뉴스 본문
out<-m$find('{"news_date":{"$gte":"2017-01-01", "$lt":"2017-12-31"}}')

library(KoNLP)
library(tm)

useSejongDic() 

options(mc.cores=1)

text <- readLines(out, encoding="UTF-8") #1.텍스트읽어오기
doc <- Corpus(VectorSource(text)) #2.말뭉치변환

doc <- TermDocumentMatrix(doc,  #3.TermDocumentMatrix변환
                          control=list(                               
                            tokenize=words,                       
                            removeNumbers=T,                     
                            removePunctuation=T,                
                            wordLengths=c(1, 5)                 
                          ))
doc <- as.matrix(doc) #4.Matrix로변환

doc <- rowSums(doc)#5.행의합(총빈도)

doc <- doc[order(doc, decreasing=T)]#빈도역순정렬

as.data.frame(doc[1:20])#상위20개단어보기