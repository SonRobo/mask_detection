videoForm.onsubmit = async (e) => {
  e.preventDefault(); // Prevent the default form submission

  const formData = new FormData(videoForm);
  const videoSource = document.getElementById("video-source");

  try {
    const response = await fetch("/video_feed", {
      method: "POST",
      body: formData,
    });

    if (response.ok) {
      // Set the video source to the streaming endpoint
      videoSource.src = "/video_stream"; // Assuming video_source is an <img> element
      videoSource.play(); // Reload the video element to start streaming
    } else {
      const data = await response.json();
      console.error("Error:", data.error); // Log any error message
    }
  } catch (error) {
    console.error("Fetch error:", error);
  }
};
