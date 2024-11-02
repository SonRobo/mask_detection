const serverUrl = `${window.location.protocol}//${window.location.host}/test_api`;

document
  .getElementById("apiTestForm")
  .addEventListener("submit", async function (e) {
    e.preventDefault(); // Prevent the default form submission

    try {
      const response = await fetch(serverUrl, {
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
