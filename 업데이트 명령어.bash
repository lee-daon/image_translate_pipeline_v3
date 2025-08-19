# 1. image_translate_worker 디렉토리로 이동
cd image_translate_worker

# 2. 도커 이미지 빌드
docker build -t leedaon/image_translate_worker:v0.1.4 .

# 3. latest 태그도 추가 (선택사항)
docker tag leedaon/image_translate_worker:v0.1.4 leedaon/image_translate_worker:latest

# 4. Docker Hub에 푸시 (로그인 필요)
docker push leedaon/image_translate_worker:v0.1.4
docker push leedaon/image_translate_worker:latest