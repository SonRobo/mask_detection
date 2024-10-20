document.getElementById("videoForm").addEventListener("submit", function (e) {
  e.preventDefault();

  const formData = new FormData();
  const videoFile = document.getElementById("video").files[0];
  formData.append("video", videoFile);

  fetch("http://127.0.0.1:5000/video_detection", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.videoUrl) {
        // Show the video
        const videoSource = document.getElementById("video-source");
        const resultVideo = document.getElementById("result-video");
        videoSource.src = data.videoUrl;
        resultVideo.style.display = "block";
        resultVideo.load();
      }
    })
    .catch((error) => console.error("Error:", error));
});
