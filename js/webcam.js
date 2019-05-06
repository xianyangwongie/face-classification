var WEBCAM = document.getElementById('webcam');
var FACE_FRAME = document.getElementById('webcam_frame');
var CANVAS_FRAME = document.getElementById('canvas_webcam');
var CANVAS_FRAME_CTX = CANVAS_FRAME.getContext('2d');

var CANVAS_FACE = document.getElementById('face_profile_chart');
var CANVAS_FACE_CTX = CANVAS_FACE.getContext('2d');
var CANVAS_FACE_GREY = document.getElementById('face_profile_greyscale');
var CANVAS_FACE_GREY_CTX = CANVAS_FACE_GREY.getContext('2d');


var CANVAS_FACE_SMALL = document.getElementById('face_profile_chart_small');
var CANVAS_FACE_CTX_SMALL = CANVAS_FACE_SMALL.getContext('2d');
var CANVAS_FACE_GREY_SMALL = document.getElementById('face_profile_greyscale_small');
var CANVAS_FACE_GREY_CTX_SMALL = CANVAS_FACE_GREY_SMALL.getContext('2d');


// var EXPAND_BOX = {'x': 0, 'y': 0, 'w': 0, 'h': 0};
var EXPAND_BOX = {'x': -5, 'y': -5, 'w': 5, 'h': 5}
var TRACKER = new tracking.ObjectTracker(['face']);

var emotion = null
var gender = null

$(document).ready(function() {
    tracking.track('#webcam', TRACKER, { camera: true });
    setTimeout(function(){ setCanvasFrameSize(); }, 3000);
});



TRACKER.on('track', function(faces) {

    CANVAS_FRAME_CTX.clearRect(0, 0, CANVAS_FRAME.width, CANVAS_FRAME.height);

    if (faces.data.length == 0
        || IS_MODEL_GENDER_LOADED == false
        || IS_MODEL_EMOTION_LOADED == false) {
        return;
    }

    var rect = faces.data[0];

    rect.x = rect.x - EXPAND_BOX.x;
    rect.y = rect.y - EXPAND_BOX.y;
    rect.width = rect.width + EXPAND_BOX.w;
    rect.height = rect.height + EXPAND_BOX.h;

    drawFaceFrame(rect);
    cropFace(rect);

    var result_gender_age = getResultGenderAge(CANVAS_FACE_GREY);
    var result_emotion = getResultEmotion(CANVAS_FACE_GREY_SMALL);
 
    $('#age').text(result_gender_age.age.label)
    if(result_gender_age.gender.percent >= 70 ){
        $('#gender').text(result_gender_age.gender.label)
    }
    $('#emotion').text(result_emotion.label)
});

function setCanvasFrameSize() {
    var w = $('#webcam').width();
    var h = $('#webcam').height();
    CANVAS_FRAME.width = w;
    CANVAS_FRAME.height = h;
}

function drawFaceFrame(rect) {
    CANVAS_FRAME_CTX.strokeStyle = '#a64ceb';
    CANVAS_FRAME_CTX.strokeRect(rect.x, rect.y, rect.width, rect.height);
}

function cropFace(rect) {

    var x = rect.x
    var y = rect.y;
    var w = rect.width;
    var h = rect.height;

    var w_w = $(WEBCAM).width();
    var w_h = $(WEBCAM).height();
    var video_w = WEBCAM.videoWidth;
    var video_h = WEBCAM.videoHeight;

    var ratio = video_w / w_w;

    CANVAS_FACE_CTX.drawImage(WEBCAM, x*ratio, y*ratio, w * ratio, h * ratio, 0, 0, 224, 224);
    CANVAS_FACE_CTX_SMALL.drawImage(WEBCAM, x*ratio, y*ratio, w * ratio, h * ratio, 0, 0, 64, 64);

    //Convert Image to Greyscale
    var imageData = CANVAS_FACE_CTX.getImageData(0, 0, CANVAS_FACE.width, CANVAS_FACE.height);
    var data = imageData.data;
    for (var i = 0; i < data.length; i += 4) {
        var avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
        data[i]     = avg; // red
        data[i + 1] = avg; // green
        data[i + 2] = avg; // blue
    }
    CANVAS_FACE_GREY_CTX.putImageData(imageData, 0, 0);

    //Convert Small Image to Greyscale
    var imageData_small = CANVAS_FACE_CTX_SMALL.getImageData(0, 0, CANVAS_FACE_SMALL.width, CANVAS_FACE_SMALL.height);
    var data = imageData_small.data;
    for (var i = 0; i < data.length; i += 4) {
        var avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
        data[i]     = avg; // red
        data[i + 1] = avg; // green
        data[i + 2] = avg; // blue
    }
    CANVAS_FACE_GREY_CTX_SMALL.putImageData(imageData_small, 0, 0);
}


function getResultGenderAge(im) {
    var input = preprocess_input(im);
    var result = predictGender(input);
    return result;
}

function getResultEmotion(im) {
    var input = preprocess_input_small(im);
    var result = predictEmotion(input);
    return result;
}

