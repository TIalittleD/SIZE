# 使用 tiangolo/uwsgi-nginx-flask 镜像作为基础镜像
FROM tiangolo/uwsgi-nginx-flask:python3.9

# 将当前目录下的 app 目录复制到镜像的 /app 目录中
COPY ./app /app

# 安装依赖
RUN pip install --no-cache-dir -r /app/tmp/requirements.txt
