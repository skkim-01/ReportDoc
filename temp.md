## react + chrome extension issues

##### 1. create-react-app으로 리액트 프로젝트 생성
##### 2. manifest.json을 chrome extension 포맷에 맞게 수정

##### 3. Extension 동작 확인 완료
![image](https://user-images.githubusercontent.com/46100421/159676227-d837dea7-703a-4eef-986b-2b67f3677900.png)

![image](https://user-images.githubusercontent.com/46100421/159675226-a9fc8c95-e351-47b1-a9f1-7a9f847dd241.png)

##### 4.1. bip39 패키지 사용 시 다음 에러 확인
```javascript
const bip39 = require('bip39')
const mnemonic = bip39.generateMnemonic()
```

![image](https://user-images.githubusercontent.com/46100421/159675182-fb6f5120-96c6-4033-86e5-cbdd7dbd424e.png)

##### 4.2. 웹팩 이슈 또한 확인
- 텍스트를 저장하지 못하였으나, buffer 패키지도 동일한 이슈 발생
```
Creating an optimized production build...
Failed to compile.

Module not found: Error: Can't resolve 'stream' in 'D:\project\publish-wallet-extension\node_modules\cipher-base'
BREAKING CHANGE: webpack < 5 used to include polyfills for node.js core modules by default.
This is no longer the case. Verify if you need this module and configure a polyfill for it.

If you want to include a polyfill, you need to:
        - add a fallback 'resolve.fallback: { "stream": require.resolve("stream-browserify") }'
        - install 'stream-browserify'
If you don't want to include a polyfill, you can use an empty module like this:
        resolve.fallback: { "stream": false }


npm ERR! code ELIFECYCLE
npm ERR! errno 1
npm ERR! publish-wallet-extension@0.1.0 build: `react-scripts build`
npm ERR! Exit status 1
```

##### 5. 웹팩 설정의 resolve 수정
- ./node_modules/react-scripts/config/webpack.config.js
- fallback 추가 및 아래 두 패키지 설치
```javascript
resolve: {
      fallback: { 
        "stream": require.resolve("stream-browserify"),
        "buffer": require.resolve("buffer")
      },
```

##### 6. bip39 패키지 내부에 있는 buffer 패키지 이슈 발생
![image](https://user-images.githubusercontent.com/46100421/159676456-43fcddf3-70d3-4a13-be77-28019957d714.png)

##### 7. package.json
```json
{
  "name": "extension test",
  "version": "0.1.0",
  "private": true,
  "dependencies": {
    "@testing-library/jest-dom": "^5.16.2",
    "@testing-library/react": "^12.1.4",
    "@testing-library/user-event": "^13.5.0",
    "axios": "^0.26.1",
    "bip39": "^3.0.4",
    "buffer": "^6.0.3",
    "ethers": "^5.6.1",
    "react": "^17.0.2",
    "react-dom": "^17.0.2",
    "react-scripts": "5.0.0",
    "stream-browserify": "^3.0.0",
    "web-vitals": "^2.1.4"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "eslintConfig": {
    "extends": [
      "react-app",
      "react-app/jest"
    ]
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  },
  "devDependencies": {
    "react-app-rewired": "^2.2.1"
  }
}
```
