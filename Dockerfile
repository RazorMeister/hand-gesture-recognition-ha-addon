# Use an official Python runtime as a parent image
FROM python:3.9

# Install libgl1 to resolve libGL.so.1 dependency (libgl1-mesa-glx removed in Debian Trixie)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*


# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --prefer-binary --timeout 120 -r requirements.txt

# Run script.py when the container launches
CMD ["python", "script.py"]
