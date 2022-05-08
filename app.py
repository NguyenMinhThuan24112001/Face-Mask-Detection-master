from flask import Flask, render_template, request, send_from_directory
from DetectMask import detectMask
import cv2

COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1
detectMask = detectMask()

@app.route('/')
def man():
    return render_template('index.html')

@app.route('/home', methods=['POST'])
def home():
    global COUNT
    img = request.files['image']

    img.save('static/{}.jpg'.format(COUNT))
    filename = 'static/{}.jpg'.format(COUNT)

    label, image = detectMask.detectMaskImage(filename)
    cv2.imwrite('static/{}.jpg'.format(COUNT), image)

    preds = label
    COUNT += 1
    return render_template('prediction.html', data=preds)


@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static', "{}.jpg".format(COUNT-1))


if __name__ == '__main__':
    app.run(debug=True)



