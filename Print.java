package com.erohsik.excel;

import java.io.FileInputStream;
import java.io.FileNotFoundException;

import javax.print.Doc;
import javax.print.DocFlavor;
import javax.print.DocPrintJob;
import javax.print.PrintException;
import javax.print.PrintService;
import javax.print.PrintServiceLookup;
import javax.print.SimpleDoc;

public class Print {

	public static void main(String[] args) {
		
		FileInputStream psStream = null;  
	    try {  
	        psStream = new FileInputStream("c:\\test.pdf");  
	        } catch (FileNotFoundException ffne) {  
	          ffne.printStackTrace();  
	        }  
	    if (psStream == null) {  
	        return;  
	    }  
	    
	    
		DocFlavor f = DocFlavor.INPUT_STREAM.AUTOSENSE;
		PrintService ps = PrintServiceLookup.lookupDefaultPrintService();
		PrintService pServices[] = PrintServiceLookup.lookupPrintServices(f, null);
		System.out.println("DEFAULT:"+ps.getName());
		for(PrintService p : pServices) {
			System.out.println("-->:"+p.getName());
		}
		
		DocFlavor docs[] = ps.getSupportedDocFlavors();
		for (DocFlavor docFlavor : docs) {
			System.out.println("doc->" + docFlavor.toString());
		}
		
		Doc document = new SimpleDoc(psStream, f, null);
		DocPrintJob job = ps.createPrintJob();
		try {
			job.print(document, null);
		} catch (PrintException e) {
			e.printStackTrace();
		}
	}
}

//https://stackoverflow.com/questions/1407459/silent-printing-of-pdf-from-within-java