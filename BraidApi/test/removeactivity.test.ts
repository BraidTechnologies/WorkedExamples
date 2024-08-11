'use strict';
// Copyright Braid Technologies Ltd, 2024


import { expect } from 'expect';
import { describe, it } from 'mocha';

import { EEnvironment } from '../../BraidCommon/src/IEnvironment';
import { ActivityRepostoryApi} from '../../BraidCommon/src/ActivityRepositoryApi'


declare var process: any;

function randomInt(min : number, max: number) : number {
   return Math.floor(Math.random() * (max - min)) + min;
}

function randomKey () : string {
   return randomInt (0, 1000000000).toString();
}

describe("RemoveActivity", async function () {

   let record = {
      storeId: randomKey(),
      storeTimestamp: new Date(),
      storeContextId: "madeupId",
      storeUserId: "madeeupId",
      storeClassName: "madeUpClass",
      test: "Some test data"      
   }
      
   it("Needs to succeed with valid key in local environment", async function () {
      
      let api = new ActivityRepostoryApi (EEnvironment.kLocal, process.env.SessionKey.toString());

      let myRecord = { ...record };
      myRecord.storeId = randomKey();

      let notremoved = await api.remove (myRecord);     
      let stored = await api.save (myRecord); 
      let removed = await api.remove (myRecord);       

      expect (notremoved).toBe (false) ;       
      expect (stored).toBe (true) ;         
      expect (removed).toBe (true) ;     

   }).timeout(20000);

   it("Needs to succeed with valid key in production environment", async function () {
      
      let api = new ActivityRepostoryApi (EEnvironment.kProduction, process.env.SessionKey.toString());

      let myRecord = { ...record };
      myRecord.storeId = randomKey();

      let notremoved = await api.remove (myRecord);     
      let stored = await api.save (myRecord); 
      let removed = await api.remove (myRecord);       

      expect (notremoved).toBe (false) ;       
      expect (stored).toBe (true) ;         
      expect (removed).toBe (true) ;   

   }).timeout(20000);

   it("Needs to fail with invalid key.", async function () {

      let api = new ActivityRepostoryApi (EEnvironment.kLocal, "thiswillfail");

      let myRecord = { ...record };
      myRecord.storeId = randomKey();

      let ok = await api.remove (myRecord); 

      expect (ok).toBe (false) ;        

   }).timeout(20000);

   it("Needs to fail with invalid key in production environment.", async function () {

      let api = new ActivityRepostoryApi (EEnvironment.kLocal, "thiswillfail");

      let myRecord = { ...record };
      myRecord.storeId = randomKey();

      let ok = await api.remove (myRecord);    

      expect (ok).toBe (false) ;              

      }).timeout(20000);

});