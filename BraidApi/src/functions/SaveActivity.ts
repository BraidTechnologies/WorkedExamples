'use strict';
// Copyright Braid Technologies Ltd, 2024
// 'func azure functionapp publish BraidApi' to publish to Azure 
// 'npm start' to run locally

// 3rd party imports
import { app, HttpRequest, HttpResponseInit, InvocationContext } from "@azure/functions";
import axios from "axios";

// Internal imports
import { defaultOkResponse, isSessionValid, sessionFailResponse } from "./Utility";
import { throwIfUndefined } from "../../../BraidCommon/src/Asserts";
import { IStorable } from "../../../BraidCommon/src/IStorable";
import { defaultPartitionKey, makePostActivityToken, makePostActivityHeader } from './CosmosRepositoryApi';


/**
 * Saves an activity record based on the provided request and context.
 * Validates the session key from the request query parameters against predefined session keys.
 * If the session key is valid, logs the validation status, processes the JSON request, and saves the activity.
 * Returns an HTTP response with a status code and the session key or an error message.
 *
 * @param request - The HTTP request containing the activity data.
 * @param context - The context for the current invocation.
 * @returns A promise that resolves to an HTTP response with the status and response body.
 */
export async function saveActivity(request: HttpRequest, context: InvocationContext): Promise<HttpResponseInit> {


   if (isSessionValid(request, context)) {

      let jsonRequest: IStorable = await request.json() as IStorable;

      try {
         await saveActivityDb(jsonRequest, context);
         context.log("Saved:" + jsonRequest.toString());
      }
      catch (e: any) {
         context.error("Failed save:" + e.toString());
         return {
            status: 500,
            body: "Failed save."
         };
      }

      return defaultOkResponse();
   }
   else {
      return sessionFailResponse();
   }
};

app.http('SaveActivity', {
   methods: ['POST'],
   authLevel: 'anonymous',
   handler: saveActivity
});

async function saveActivityDb(record: IStorable, context: InvocationContext): Promise<boolean> {

   let dbkey = process.env.CosmosApiKey;

   let done = new Promise<boolean>(function (resolve, reject) {

      let time = new Date().toUTCString();
      let stream = JSON.stringify(record);
      let document = JSON.parse(stream);

      throwIfUndefined(dbkey); // Keep compiler happy, should not be able to get here with actual undefined key. 
      let key = makePostActivityToken(time, dbkey as string);
      let headers = makePostActivityHeader(key, time, defaultPartitionKey);

      document.partition = defaultPartitionKey; // Dont need real partitions until 10 GB ... 

      axios.post('https://braidstudio.documents.azure.com:443/dbs/Studio/colls/Activity/docs/',
         document,
         {
            headers: headers
         })
         .then((resp: any) => {

            resolve(true);
         })
         .catch((error: any) => {

            context.log("Error calling database:", error);
            reject(false);
         });
   });

   return done;
}

