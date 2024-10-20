document
  .getElementById("apiTestForm")
  .addEventListener("submit", async function (e) {
    e.preventDefault(); // Prevent the default form submission

    try {
      const response = await fetch("http://127.0.0.1:5000/test_api", {
        method: "GET",
      });

      if (response.ok) {
        const data = await response.json();
        alert(data.msg); // Display the message returned by the API
      } else {
        console.error("Error calling API");
      }
    } catch (error) {
      console.error("Error:", error);
    }
  });
