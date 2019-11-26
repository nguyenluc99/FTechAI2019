import scrapy
import re
import csv
from scrapy_splash import SplashRequest



class Q1Spider(scrapy.Spider):
    name = "q1"
    start_urls = [
                'https://tiki.vn/nha-gia-kim-p378448.html',
                # 'https://tiki.vn/5-centimet-tren-giay-p418676.html',
                # 'https://tiki.vn/tony-buoi-sang-tren-duong-bang-tai-ban-2017-p1005148.html',
                ]

    def parse(self, response):
        print('response is ', response.css('.special-price-item').extract_first())
        for price in response.css('.special-price-item').css('#span-price') :
            print('================== price is : ',re.sub("[^0-9]","",price.css('span ::text').extract_first()))
            
        for name in response.css('.no-js').css('.item-name'):#.css('wrap').css('.container product-container').css('product-cart').css('item-box').css('item-name'):
            print('================== name is : ', name.css('span ::text').extract_first())
        
        for author in response.css('.no-js').css('.info').css('.item-brand'):#.css('wrap').css('.container product-container').css('product-cart').css('item-box').css('item-name'):
            print('================== author is : ', author.css('p ::text').extract_first())
        
        for intro in response.css('.no-js').css('.product-content-detail').css('#gioi-thieu') : 
            print('================== intro is : ', '\n'.join(intro.css('p ::text').extract()))
        
        print('================== image is : ', response.css('.no-js').css('.product-magiczoom').css('img').attrib['src'])


class Q2Spider(scrapy.Spider) :
    name = "q2"
    numOfPage = 5
    res = 0
    content = [['title', 'author', 'price']]
    def start_requests(self):
        sampleURL = 'https://tiki.vn/sach-van-hoc/c839'
        for i in range(0, self.numOfPage ):
            yield scrapy.Request(sampleURL + '?page=' + str(i + 1), self.parse)
        
    def parse(self, response) :
        for book in response.css('.product-box-list').css('.product-item    '):
            title = str(book.css('div').attrib['data-title'])
            author = str(book.css('div').attrib['data-brand'])
            price = str(book.css('div').attrib['data-price'])
            self.content.append([title, author, price])
        self.res += 1
        if (self.res == self.numOfPage): self.writeCSV()
    def writeCSV(self) :
        with open('dataQ2.csv', 'w') as outputFile :
            writer = csv.writer(outputFile, delimiter = ' ', quotechar = '"', quoting=csv.QUOTE_MINIMAL)
            for el in self.content :
                writer.writerow(el)

class Q3Spider(scrapy.Spider) :
    name = "q3"
    numOfPage = 40
    res = 0
    content = [['title', 'price']]
    def start_requests(self):
        sampleURL = 'https://www.fahasa.com/sach-trong-nuoc/van-hoc-trong-nuoc/page/'
        for i in range(0, self.numOfPage ):
            yield SplashRequest(sampleURL + str(i + 1) + '.html', self.parse, args = {"wait": 5})
        
    def parse(self, response) :
        print('====================================================')
        lst = response.css('#offcanvas-container').css('.ma-box-content')
        for item in lst :
            title = item.css('.product-name').css('a ::text').extract_first()
            price = re.sub("[^0-9]","",(item.css('.clearfix').css('.price').css('span ::text').extract_first()))
            self.content.append([title, price])
        self.res += 1
        if (self.res == self.numOfPage): self.writeCSV()

    def writeCSV(self) :
        with open('dataQ3.csv', 'w') as outputFile :
            writer = csv.writer(outputFile, delimiter = ' ', quotechar = '"', quoting=csv.QUOTE_MINIMAL)
            for el in self.content : 
                writer.writerow(el)

# class Q3ImageSpider(scrapy.Spider) :
#     name = "q3Image"
#     def start_requests(self):
#         sampleURL = 'https://www.google.com/search?q=món+ăn+việt+nam&sxsrf=ACYBGNQKUzjR9NJXITJVlblIEgG3RmLxdA:1569494044766&source=lnms&tbm=isch&sa=X&ved=0ahUKEwjxq4_4pO7kAhVLZt4KHUmPBJ4Q_AUIEigB&biw=1296&bih=669'
#         yield scrapy.Request(sampleURL, self.parse)


#         # return super().start_requests()

#     def parse(self, response) :
        
        
        
#         return 1
