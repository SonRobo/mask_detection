document
  .getElementById("imageForm")
  .addEventListener("submit", async function (e) {
    e.preventDefault(); // Prevent the form from reloading the page

    const formData = new FormData(); // Create FormData to hold the file
    const fileField = document.querySelector('input[type="file"]');

    formData.append("img", fileField.files[0]); // Append the selected file

    try {
      const response = await fetch("http://127.0.0.1:5000/mask_detection", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        const imgUrl = data.imageUrl; // Assuming the API returns the image URL in 'imageUrl' field

        // Set the returned image URL to the 'src' attribute of the image tag
        document.getElementById("result-image").src = imgUrl;
      } else {
        console.error("Error uploading image");
      }
    } catch (error) {
      console.error("Error:", error);
    }
  });
