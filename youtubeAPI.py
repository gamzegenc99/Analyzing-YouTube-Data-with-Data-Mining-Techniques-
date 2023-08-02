from googleapiclient.discovery import build
import pandas as pd #for creating data frame


videoId ="HX8-QxHHTJQ"
next_page_token = "QURTSl9pMmxOc1FSOVFUWVBEVGN0YzNtRDNlVVlqVkIzMmw5MGxxVDVwTWVVT002Q0xSQlprY25jX090bXJDd0NrVDY1TGpuSVRwRmJINE1XUWhJX0hGMzFDbWt3VFk1"

def get_youtube(): #create an instance of the api
    DEVELOPER_KEY ='AIzaSyDwL9D9aEeCL-6_bb9c5DEbTHn3UbERog4'
    YOUTUBE_API_SERVICE_NAME ='youtube'
    YOUTUBE_API_VERSION = 'v3'
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey= DEVELOPER_KEY, cache_discovery=False)
    return youtube    #timeout = 1

youtube = get_youtube() 
request = youtube.commentThreads().list(
                      part="snippet",  
                      #type='video',
                      videoId = videoId,
                      maxResults=100,
                      pageToken = next_page_token
                  )
response = request.execute()

data=[]
#while True:
   
video_ids=[]
for _ in range(20):# 500 bintane yorum almak için range 5 sınırı koydum.her sayfa için 100 tane yorum çekiyor bnu yüzden 30 sayfa için 500 yorum çekecek.

                   
                    video_ids.append(videoId),
                    # İstek parametrelerini güncellenecek.

                    request= youtube.commentThreads().list(
                    part = "snippet",#From the answer of commentThreads,have to add the replies parameter in order to retrieve the replies the comments might have.
                    videoId = videoId ,
                    maxResults=100,
                    pageToken = next_page_token
                    )
                    response = request.execute()
                    
                    request = youtube.videos().list(
                            part="snippet,statistics",
                            id=videoId,
                            maxResults=100
                    )
                    response1 = request.execute()
                    
                    for item in (response1["items"]): #request sonucunda gelen response içerisinde dolaşarak verileri alıyorum.
                        video_publishedAt = item["snippet"]["publishedAt"]
                        videoViewCount =item["statistics"]["viewCount"]
                        videoLikeCount =item["statistics"]["likeCount"]
                        videoCommentCount = item["statistics"]["commentCount"]
                        title=item["snippet"]["title"]
            
                
                    for item in (response["items"]): #request sonucunda gelen response içerisinde dolaşarak verileri alıyorum.
                        comment = item["snippet"]["topLevelComment"]
                        author= comment["snippet"]["authorDisplayName"]
                        author_channel_url=comment["snippet"]["authorChannelUrl"]
                        author_channel_id=comment["snippet"]["authorChannelId"]["value"]
                        like_count = comment["snippet"]["likeCount"] #The total number of likes (positive ratings) the comment has received.
                        comment_rating=comment["snippet"]["viewerRating"] #like or none
                        published_dateTime= comment["snippet"]["publishedAt"]# pdate yeni ekledim
                        comment_text = comment["snippet"]["textDisplay"]
                       
                        #geriye replies yorumları nasıl yazdırırım o kaldı ve de limit nasıl belirlerim yorum için
                        #print(author,author_channel_url,author_channel_id,like_count,comment_rating, comment_text)
                      
                       
                        #Create data frame
                        #data.append([item.author,item.author_channel_url, item.author_channel_id, item.like_count,item.comment_rating,item.comment_text])
                        data.append([title,video_publishedAt,videoViewCount,videoLikeCount,videoCommentCount,author,author_channel_url,author_channel_id,like_count,comment_rating,published_dateTime,comment_text])
                        
                    #next_page_token = response.get('nextPageToken')
                    
                        next_page_token = response.get('nextPageToken')
                    columns =['VideoTitle','VideoPublishedDate','VideoViewCount','VideoLikeCount','VideoCommentCount','UserName','UserChannelUrl','UserChannelId','LikeCount','CommentRating','DateTime','Comment']
                    df =pd.DataFrame(data, columns= columns)
                    #df.to_excel('youtubedemetakalındeneme1.xlsx')  #saving data to excel with pandas   
                    #print(df)
                    
                    # nextPageToken olup olmadığını kontrol et
    
                    if "nextPageToken" in response:
                            page_token = response["nextPageToken"]
                    else:
                            done = True
                            break             
        
               
        
df.to_excel('SeymaSubası.xlsx')                    
    
    
         
    