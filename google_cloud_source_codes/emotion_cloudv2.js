const functions = require('@google-cloud/functions-framework');
 
// Variable to store the latest cloud event data
let latestEventData = null;

const { Storage } = require('@google-cloud/storage');
 
/* packages to include in packages.json:

{
  "dependencies": {
    "@google-cloud/functions-framework": "^3.0.0",
    "@google-cloud/storage": "^5.8.0",
    "@google-cloud/firestore": "^4.9.6"
  }
}



*/

// Instantiate a storage client
const storage = new Storage();
const bucketName = 'kfupmmx'; // replace with your bucket name
const fileName = 'latestEvent.json'; // file name to store the latest event data

functions.cloudEvent('helloGCS', async cloudEvent => {
  const eventData = {
    eventId: cloudEvent.id,
    eventType: cloudEvent.type,
    bucket: cloudEvent.data.bucket,
    fileName: cloudEvent.data.name,
    metageneration: cloudEvent.data.metageneration,
    createdTime: cloudEvent.data.timeCreated,
    updatedTime: cloudEvent.data.updated
  };
 
 fetch('https://citra-eva2ckcawq-uc.a.run.app/detect', {     method: 'POST',     headers: {         'Content-Type': 'application/json'    },     body: JSON.stringify({ image_name: cloudEvent.data.name }) }) .then(response => response.json()) .then(data => console.log(data)) .catch(error => console.error('Error:', error));
  try {
    if (eventData.fileName == "latestEvent.json"){
      console.log("------------- JSON Loop Avoided ---------");
    }
    else {
      const file = storage.bucket(bucketName).file(fileName);
      const [data] = await file.download();
      console.log("------------- Data ---------", data);
      console.log("------------- Type of Data ---------", typeof(data));
      let events = JSON.parse(data.toString());
  
      // Ensure that events is an array
      if (!Array.isArray(events)) {
        events = [];
      }
  
      events.push(eventData);
      await file.save(JSON.stringify(events));
    }
  } catch (error) {
    // If the file doesn't exist or error in reading, create a new file with the event as an array
    await storage.bucket(bucketName).file(fileName).save(JSON.stringify([eventData]));
    console.log("------------- Catch Error ---------");
  }
});
