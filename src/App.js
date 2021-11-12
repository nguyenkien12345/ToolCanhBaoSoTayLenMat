import React, {useEffect, useRef, useState} from 'react'
import './App.css';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as knnClassifier from '@tensorflow-models/knn-classifier';
import {Howl, Howler} from 'howler';
import soundURL from './assets/sound.mp3';
import { initNotifications, notify } from '@mycv/f8-notification';

// Khai báo âm nhạc
var sound = new Howl({
  src: [soundURL]
});

const NOT_TOUCH_LABEL = 'not_touch';
const TOUCHED_LABEL = 'touched';
// Training cho nó học 50 lần
const TRAINING_TIMES = 50;
const TOUCHED_CONFIDENCE = 0.8;

function App() {
  const video = useRef(); 
  const classifier = useRef(); 
  const mobilenetModule = useRef(); 
  const canPlaySound = useRef(true);
  const [touched,setTouched] = useState(false);

  const init = async () => {
    console.log('init...');
    // Có Promise là phải có await
    await setUpCamera();
    console.log("Setup Camera Success");

    // Setup mobilenet và knnClassifier
    classifier.current = knnClassifier.create();
    mobilenetModule.current = await mobilenet.load(); //load(): Load database

    console.log("Setup Done");
    // Khi nào console.log("Không chạm tay lên mặt và bấm Train 1") hiện lên thì lúc đó app mới sẵn sàng
    console.log("Không chạm tay lên mặt và bấm Train 1");

    initNotifications({ cooldown: 3000 });
  }

  // SetUp Camera. Hàm này sẽ xin quyền truy cập vào camera
  const setUpCamera = () => {
    return new Promise((resolve, reject) => {
      // Xin quyền truy cập vào camera. Tuỳ vào browser trình duyệt mà nó sẽ tương ứng với pt hỗ trợ của nó
      navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia;
      if(navigator.getUserMedia) // Nếu quyền truy cập camera thành công
      {
        navigator.getUserMedia(
          // Xin quyền video. Nếu thành công nó sẽ trả về đối tượng stream
          { video : true }, stream => {
            video.current.srcObject = stream;
            video.current.addEventListener("loadeddata",resolve); // Sự kiện này sẽ xảy ra khi mà chúng ta load thành công video
          },
          // Khi lỗi nó sẽ gọi về
          error => reject(error)
        );
      }
      else
      {
        reject();
      }
    });
  }

  const sleep = (ms = 0) => {
    return new Promise(resolve => setTimeout(resolve, ms))
  }

  // Khi click vào button sẽ thực hiện hành động train
  // Khi sử dụng await bắt buộc phải có async
  const train = async label =>{
    console.log(label);
    for (let i = 0; i < TRAINING_TIMES; ++i)
    {
      console.log(`Progress ${parseInt((i+1)/TRAINING_TIMES * 100)}% `); // Hiển thị % tiến trình train
      await training(label);
    }
  }

  // Bước 1: Train cho máy khuôn mặt không chạm tay
  // Bước 2: Train cho máy khuôn mặt có chạm tay
  // Bước 3: Lấy hình ảnh hiện tại, phân tích và so sánh với data đã học trước đó
  // ==> Nếu mà matching với data khuôn mặt chạm tay ==> Cảnh báo
  // * @param {*} label
  //  

  const training = label => {
    return new Promise(async resolve => {
      const embedding = mobilenetModule.current.infer(
        // Đẩy Frame hình ảnh của video lên đưa vào database của mobilenetModule sau đó nó xử lý trả về 1 mạng dữ liệu
        video.current,
        true
      );
      classifier.current.addExample(embedding, label); 
      await sleep(100);
      resolve();
    });
  }

  // Có promise là có async, await
  const run = async () => {
    const embedding = mobilenetModule.current.infer(
      // Đẩy Frame hình ảnh của video lên đưa vào database của mobilenetModule sau đó nó xử lý trả về 1 mạng dữ liệu
      video.current,
      true
    );
    const result = await classifier.current.predictClass(embedding);

    // console.log('Label: ', result.label);
    // console.log('Confidences: ', result.confidences); // Độ tin tưởng
    // Sau khi nghĩ 200s gọi cho nó chạy cứ thế liên tục
    if (result.label === TOUCHED_LABEL && result.confidences[result.label] > TOUCHED_CONFIDENCE)
    {
      console.log("Touched");
      if (canPlaySound.current)
      {
        canPlaySound.current = false;
        sound.play();
      }
      notify('Bỏ tay ra!', { body: 'Bạn vừa chạm tay vào mặt!' });
      setTouched(true);
    }
    else
    {
      console.log("Not Touch");
      setTouched(false);
    }
    await sleep(200);
    run();
  }

  useEffect(() => {
    init();

    // Hết Audio mới cho lặp lại.
    sound.on('end',function(){
      canPlaySound.current = true;
    })

    return () => {}
  }, [])

  return (
    <div className={`main ${touched ? 'touched' : ''}`}>
      <video
        ref = {video}
        className="video"
        autoPlay // Tự động run
      />

      <div className="control">
        {/* TH1: truòng hợp không đưa tay lên mặt */}
        <button className="btn" onClick={() => train(NOT_TOUCH_LABEL)}>Train 1</button>
        {/* TH2: truòng hợp đưa tay lên mặt */}
        <button className="btn" onClick={() => train(TOUCHED_LABEL)}>Train 2</button>
        <button className="btn" onClick={() => run()}>RUN</button>
      </div>
    </div>
  );
}

export default App;

// resolve là thành công
// reject là thất bại
 