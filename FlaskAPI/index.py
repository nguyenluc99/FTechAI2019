from flask import Flask, json, send_from_directory 
from flask_restful import Resource, Api, request, reqparse
from flask_limiter import Limiter, HEADERS
# from flask_limiter.util import get_remote_address
from PIL import Image
import cv2
import numpy as np
import werkzeug 


app = Flask(__name__)#, static_url_path='/Saved')
api = Api(app)

def get_user_key():
    # print(request.headers['user_key'])
    return request.headers['user_key']

# http://blog.luisrei.com/articles/flaskrest.html
# curl -H "user_key:$(pwd)/abc"  http://localhost:5000/ApiSample 
limiter = Limiter(app,
            key_func = get_user_key,
            default_limits = ["60 per minute"]
            )
# limiter.header_mapping = {
#     HEADERS.LIMIT : "1 per minute",
#     HEADERS.RESET : "1 per minute",
#     HEADERS.REMAINING: "1 per minute"
# }
# limiter.header_mapping[HEADERS.LIMIT] = '1 per minute'
class HelloWorld(Resource):
    def post(self):
        user_key = request.headers['user_key']
        print("ukey is ", user_key)
        # filename = request.headers['filename']
        
        # if request.files.get('img') : print("uploaded")
        # else : print("Not uploaded")
        # print('32222, ', request.files)
        parser = reqparse.RequestParser()
        parser.add_argument('img', type=werkzeug.FileStorage, location='files')
        self.args = parser.parse_args()
        print(self.args)
        print(type(self.args))
        # npimg = np.fromstring(self.args['img'].read(), np.uint8)
        # image = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
        # h, w, _ = image.shape
        # print(h, w)
        
        # print('=====filename : ', (filename))
        # print('=====filename : ', type(filename))
        # im = Image.open(filename)
        # print(im.format, im.size)

        # return {'hello' : 'world', 'header' : user_key, 'filename'  : filename}
        return {'hello' : 'world', 'header' : user_key}


api.add_resource(HelloWorld, '/ApiSample', methods = ['GET', 'POST'])

if __name__ == '__main__':
    # print('runnn')
    app.run(debug=True)