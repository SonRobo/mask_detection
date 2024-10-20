
videoForm.onsubmit = async (e) => {
  e.preventDefault(); // Prevent the default form submission

  const formData = new FormData(videoForm);
  const videoElement = document.getElementById("result-video");
  videoElement.src = "/static/processed/processed_Y2meta.app-Wear_a_Mask.mp4";
  videoElement.load();

  // try {
  //   const response = await fetch("/video_feed", {
  //     method: "POST",
  //     body: formData,
  //   });

  //   if (response.ok) {
  //     const data = await response.json();
  //     videoSource.src = data.output_file;

  //     videoElement.style.display = "block";
  //     videoElement.load();

  //     alert("Video uploaded successfully!");
  //   } else {
  //     const errorData = await response.json();
  //     alert(`Error: ${errorData.error}`);
  //   }
  // } catch (error) {
  //   console.error("Error uploading video:", error);
  //   alert("An error occurred while uploading the video.");
  // }
};
