const functions = require('@google-cloud/functions-framework');
const { Storage } = require('@google-cloud/storage');

// Instantiate a storage client
const storage = new Storage();
const bucketName = 'kfupmmx'; // replace with your bucket name
const fileName = 'latestEvent.json'; // file name to store the latest event data

/* packages to include in packages.json:

{
  "dependencies": {
    "@google-cloud/functions-framework": "^3.0.0",
    "@google-cloud/storage": "^5.8.0"
  }
}


*/

functions.http('showLatestEvent', async (req, res) => {
  if (req.method === 'GET') {
    try {
      const file = storage.bucket(bucketName).file(fileName);
      const meta1 = await storage.bucket(bucketName).file(fileName).getMetadata();
      console.log("-------- Size --------", meta1)
      const [data] = await file.download();
      const events = JSON.parse(data.toString());
 
      // Generate HTML table content for all events
      let tableContent = events.map(event => `
<tr>
<td>${event.eventId}</td>
<td>${event.eventType}</td>
<td>${event.bucket}</td>
<td>${event.fileName}</td>
<td>${event.metageneration}</td>
<td>${event.createdTime}</td>
<td>${event.updatedTime}</td>
</tr>
      `).join('');
 
      // Send HTML response with enhanced CSS
      res.status(200).send(`
<!DOCTYPE html>
<html>
<head>
<title>Cloud Function Event Log</title>
<style>
              body { 
                font-family: Arial, sans-serif; 
                margin: 40px; 
                background-color: #f4f4f4;
                color: #333;
              }
              h1 { 
                text-align: center; 
                color: #333;
              }
              table { 
                width: 100%; 
                border-collapse: collapse; 
                margin-bottom: 40px; 
                box-shadow: 0 2px 15px rgba(0, 0, 0, 0.1);
              }
              th, td { 
                border: 1px solid #ddd; 
                padding: 12px 15px; 
                text-align: left; 
              }
              th { 
                background-color: #007bff; 
                color: white; 
              }
              tr:nth-child(even) { 
                background-color: #f2f2f2; 
              }
              tr:hover { 
                background-color: #ddd; 
              }
              @media screen and (max-width: 600px) {
                table, thead, tbody, th, td, tr { 
                  display: block; 
                }
                thead tr { 
                  position: absolute; 
                  top: -9999px; 
                  left: -9999px; 
                }
                tr { border: 1px solid #ccc; }
                td { 
                  border: none;
                  border-bottom: 1px solid #eee; 
                  position: relative; 
                  padding-left: 50%; 
                  text-align: left;
                }
                td:before { 
                  position: absolute; 
                  top: 12px; 
                  left: 6px;
                  width: 45%; 
                  padding-right: 10px; 
                  white-space: nowrap;
                  content: attr(data-label); 
                  font-weight: bold;
                }
              }
</style>
</head>
<body>
<h1>Latest Cloud Event Data</h1>
<table>
<thead>
<tr>
<th>Event ID</th>
<th>Event Type</th>
<th>Bucket</th>
<th>File Name</th>
<th>Metageneration</th>
<th>Created Time</th>
<th>Updated Time</th>
</tr>
</thead>
<tbody>
                ${tableContent}
</tbody>
</table>
</body>
</html>
      `);
    } catch (error) {
      res.status(500).send(`Error retrieving event data: ${error.message}`);
    }
  } else {
    res.status(405).send('Method Not Allowed');
  }
});

