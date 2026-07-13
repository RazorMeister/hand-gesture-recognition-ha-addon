# Slim base -> smaller image, faster build + pull on the Pi.
FROM python:3.9-slim

# Runtime libs the wheels need on slim: libgl1/libglib2.0-0 for OpenCV,
# libgomp1 (OpenMP) for MediaPipe/NumPy, ffmpeg for RTSP decoding.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*


# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --prefer-binary --timeout 120 -r requirements.txt

# Run script.py when the container launches
CMD ["python", "script.py"]
