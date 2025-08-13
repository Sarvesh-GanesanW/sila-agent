import json
import boto3
import base64
import re
import pandas as pd
import psycopg2
import os
from datetime import datetime
from botocore.exceptions import ClientError
from io import BytesIO
import logging
import fitz
from PIL import Image
import tempfile

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class InvoiceProcessor:
    def __init__(self):
        self.s3Client = boto3.client('s3')
        self.bedrockClient = boto3.client('bedrock-runtime', region_name='us-east-1')
        self.textractClient = boto3.client('textract', region_name='us-east-1')
        
    def lambdaHandler(self, event, context):
        try:
            logger.info(f"Starting invoice extraction - Request ID: {context.aws_request_id}")
            
            s3Bucket = event.get('s3_bucket')
            s3Key = event.get('s3_key')
            pdfData = event.get('pdf_data')
            customPrompt = event.get('prompt')
            
            if not s3Bucket and not pdfData:
                return {
                    'statusCode': 400,
                    'body': {
                        'error': 'Either s3_bucket+s3_key or pdf_data is required',
                        'usage': {
                            's3_mode': 'Provide s3_bucket and s3_key',
                            'direct_mode': 'Provide pdf_data as base64'
                        }
                    }
                }
            
            if s3Bucket and s3Key:
                logger.info(f"Downloading PDF from S3: s3://{s3Bucket}/{s3Key}")
                pdfData = self.downloadPdfFromS3(s3Bucket, s3Key)
                if not pdfData:
                    return {
                        'statusCode': 400,
                        'body': {
                            'error': 'Failed to download PDF from S3',
                            'details': 'Check if file exists and Lambda has S3 permissions'
                        }
                    }
            
            pdfData = self.cleanBase64Data(pdfData)
            if not pdfData:
                return {
                    'statusCode': 400,
                    'body': {
                        'error': 'Invalid or corrupted PDF data',
                        'details': 'PDF data must be valid base64 encoded'
                    }
                }
            
            try:
                decodedPdf = base64.b64decode(pdfData)
                if not decodedPdf.startswith(b'%PDF-'):
                    return {
                        'statusCode': 400,
                        'body': {
                            'error': 'File is not a valid PDF format',
                            'detected_header': str(decodedPdf[:20]),
                            'expected_header': '%PDF-'
                        }
                    }
                logger.info(f"Valid PDF detected - Size: {len(decodedPdf)} bytes")
            except Exception as e:
                return {
                    'statusCode': 400,
                    'body': {
                        'error': 'Cannot decode PDF data',
                        'message': str(e)
                    }
                }
            
            logger.info("Processing PDF with enhanced extraction pipeline")
            extractedData = self.enhancedProcessPdf(pdfData, decodedPdf, customPrompt)
            
            dataProcessor = DataProcessor()
            excelProcessor = ExcelProcessor(self.s3Client)
            dbProcessor = DatabaseProcessor()
            
            dataFrame = dataProcessor.createDataFrame(extractedData)
            excelKey = excelProcessor.uploadToS3(dataFrame, s3Bucket)
            dbProcessor.insertToPostgres(dataFrame)
            
            processingMetadata = {
                'processed_at': datetime.now().isoformat(),
                'request_id': context.aws_request_id,
                'file_size_bytes': len(decodedPdf),
                'source_location': f"s3://{s3Bucket}/{s3Key}" if s3Bucket and s3Key else "direct_upload",
                'lambda_function': context.function_name,
                'lambda_version': context.function_version,
                'excel_location': f"s3://{s3Bucket}/{excelKey}"
            }
            
            logger.info("Invoice extraction completed successfully")
            logger.info(f"Extracted invoice: {extractedData.get('invoice_number', 'Unknown')} from {extractedData.get('vendor_name', 'Unknown vendor')}")
            
            return {
                'statusCode': 200,
                'body': {
                    'success': True,
                    'extracted_data': extractedData,
                    'processing_metadata': processingMetadata,
                    'summary': {
                        'invoice_number': extractedData.get('invoice_number'),
                        'vendor_name': extractedData.get('vendor_name'),
                        'total_amount': extractedData.get('total_amount'),
                        'currency': extractedData.get('currency'),
                        'invoice_date': extractedData.get('invoice_date'),
                        'extraction_confidence': self.calculateExtractionConfidence(extractedData)
                    }
                }
            }
            
        except ClientError as e:
            errorCode = e.response['Error']['Code']
            errorMessage = e.response['Error']['Message']
            logger.error(f"AWS Client Error: {errorCode} - {errorMessage}")
            
            return {
                'statusCode': 500,
                'body': {
                    'error': 'AWS service error',
                    'service': 'bedrock' if 'bedrock' in errorMessage.lower() else 's3',
                    'code': errorCode,
                    'message': errorMessage,
                    'suggestion': self.getErrorSuggestion(errorCode)
                }
            }
            
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return {
                'statusCode': 500,
                'body': {
                    'error': 'Invoice extraction failed',
                    'message': str(e),
                    'request_id': context.aws_request_id
                }
            }

    def downloadPdfFromS3(self, bucket, key):
        try:
            response = self.s3Client.get_object(Bucket=bucket, Key=key)
            pdfBytes = response['Body'].read()
            
            if len(pdfBytes) > 50 * 1024 * 1024:
                logger.warning(f"Large PDF file detected: {len(pdfBytes)} bytes")
            
            return base64.b64encode(pdfBytes).decode('utf-8')
            
        except ClientError as e:
            errorCode = e.response['Error']['Code']
            if errorCode == 'NoSuchKey':
                logger.error(f"File not found: s3://{bucket}/{key}")
            elif errorCode == 'AccessDenied':
                logger.error(f"Access denied to s3://{bucket}/{key}")
            else:
                logger.error(f"S3 error {errorCode}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected S3 error: {e}")
            return None

    def cleanBase64Data(self, data):
        if not data:
            logger.error("Empty PDF data provided")
            return None
        
        data = data.strip()
        
        if data.startswith('data:'):
            logger.info("Removing data URL prefix from base64 data")
            if ',' in data:
                data = data.split(',', 1)[1]
        
        data = re.sub(r'\s+', '', data)
        
        if not re.match(r'^[A-Za-z0-9+/]*={0,2}$', data):
            logger.error("Invalid base64 character set detected")
            return None
        
        if len(data) % 4 != 0:
            logger.error(f"Invalid base64 length: {len(data)} (must be multiple of 4)")
            return None
        
        logger.info(f"Base64 data validated successfully - Length: {len(data)}")
        return data

    def enhancedProcessPdf(self, pdfData, decodedPdf, customPrompt=None):
        try:
            logger.info("Starting enhanced PDF processing pipeline")
            
            pdfInfo = self.analyzePdfStructure(decodedPdf)
            logger.info(f"PDF analysis: {pdfInfo['pages']} pages, text quality: {pdfInfo['text_quality']}")
            
            if pdfInfo['pages'] > 1:
                logger.info(f"Multi-page PDF detected ({pdfInfo['pages']} pages)")
                return self.processMultiPagePdf(pdfData, decodedPdf, customPrompt)
            
            if pdfInfo['text_quality'] < 0.5:
                logger.info("Poor text quality detected, using OCR-enhanced extraction")
                return self.processWithOcrFallback(pdfData, decodedPdf, customPrompt)
            
            extractedData = self.processWithBedrock(pdfData, customPrompt)
            
            extractedData = self.enhanceExtractionWithTables(extractedData, decodedPdf)
            
            extractedData = self.validateAndCleanData(extractedData)
            
            confidence = self.calculateExtractionConfidence(extractedData)
            
            if confidence < 0.7:
                logger.info(f"Low confidence ({confidence}), trying enhanced extraction")
                enhancedData = self.retryWithEnhancedStrategy(pdfData, decodedPdf, extractedData, customPrompt)
                return self.validateAndCleanData(enhancedData)
            
            return extractedData
            
        except Exception as e:
            logger.error(f"Enhanced processing failed, falling back to basic: {e}")
            return self.processWithBedrock(pdfData, customPrompt)

    def analyzePdfStructure(self, pdfBytes):
        try:
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(pdfBytes)
                temp_file.flush()
                
                doc = fitz.open(temp_file.name)
                pageCount = len(doc)
                
                totalText = ""
                totalChars = 0
                textualChars = 0
                
                for pageNum in range(min(3, pageCount)):
                    page = doc[pageNum]
                    text = page.get_text()
                    totalText += text
                    totalChars += len(text)
                    textualChars += sum(1 for c in text if c.isalnum())
                
                doc.close()
                os.unlink(temp_file.name)
                
                textQuality = textualChars / max(totalChars, 1)
                
                return {
                    'pages': pageCount,
                    'text_quality': textQuality,
                    'total_chars': totalChars,
                    'has_tables': 'table' in totalText.lower() or '|' in totalText,
                    'sample_text': totalText[:200]
                }
                
        except Exception as e:
            logger.error(f"PDF analysis failed: {e}")
            return {'pages': 1, 'text_quality': 1.0, 'total_chars': 0, 'has_tables': False}

    def processMultiPagePdf(self, pdfData, pdfBytes, customPrompt=None):
        try:
            logger.info("Processing multi-page PDF with page splitting")
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(pdfBytes)
                temp_file.flush()
                
                doc = fitz.open(temp_file.name)
                allExtractions = []
                
                for pageNum in range(len(doc)):
                    logger.info(f"Processing page {pageNum + 1}/{len(doc)}")
                    
                    singlePageDoc = fitz.open()
                    singlePageDoc.insert_pdf(doc, from_page=pageNum, to_page=pageNum)
                    
                    pageBytes = singlePageDoc.write()
                    pageBase64 = base64.b64encode(pageBytes).decode('utf-8')
                    
                    pagePrompt = self.getPageSpecificPrompt(pageNum, customPrompt)
                    pageExtraction = self.processWithBedrock(pageBase64, pagePrompt)
                    
                    if pageExtraction:
                        allExtractions.append({
                            'page': pageNum + 1,
                            'data': pageExtraction,
                            'confidence': self.calculateExtractionConfidence(pageExtraction)
                        })
                    
                    singlePageDoc.close()
                
                doc.close()
                os.unlink(temp_file.name)
                
                return self.mergePageExtractions(allExtractions)
                
        except Exception as e:
            logger.error(f"Multi-page processing failed: {e}")
            return self.processWithBedrock(pdfData, customPrompt)

    def processWithOcrFallback(self, pdfData, pdfBytes, customPrompt=None):
        try:
            logger.info("Using Textract OCR for scanned/poor quality PDF")
            
            response = self.textractClient.analyze_document(
                Document={'Bytes': pdfBytes},
                FeatureTypes=['TABLES', 'FORMS']
            )
            
            extractedText = self.parseTextractResponse(response)
            
            if len(extractedText) > 100:
                logger.info(f"OCR extracted {len(extractedText)} characters, processing with Bedrock")
                ocrPrompt = self.getOcrEnhancedPrompt(extractedText, customPrompt)
                return self.processTextWithBedrock(extractedText, ocrPrompt)
            else:
                logger.warning("OCR extracted minimal text, falling back to direct PDF processing")
                return self.processWithBedrock(pdfData, customPrompt)
                
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            return self.processWithBedrock(pdfData, customPrompt)

    def retryWithEnhancedStrategy(self, pdfData, pdfBytes, originalData, customPrompt=None):
        try:
            logger.info("Retrying with enhanced extraction strategy")
            
            enhancedPrompt = self.getEnhancedExtractionPrompt(originalData, customPrompt)
            retryResult = self.processWithBedrock(pdfData, enhancedPrompt)
            
            if self.calculateExtractionConfidence(retryResult) > self.calculateExtractionConfidence(originalData):
                logger.info("Enhanced strategy improved extraction quality")
                return retryResult
            
            logger.info("Trying OCR-enhanced extraction as final attempt")
            return self.processWithOcrFallback(pdfData, pdfBytes, customPrompt)
            
        except Exception as e:
            logger.error(f"Enhanced retry failed: {e}")
            return originalData

    def getPageSpecificPrompt(self, pageNum, customPrompt=None):
        if customPrompt:
            return f"This is page {pageNum + 1} of a multi-page document. {customPrompt}"
        
        if pageNum == 0:
            return self.getFirstPagePrompt()
        else:
            return self.getSubsequentPagePrompt()

    def getFirstPagePrompt(self):
        return """This is the first page of an invoice document. Focus on extracting:
1. Invoice header information (number, date, vendor details)
2. Billing and shipping addresses
3. Invoice totals and payment terms
4. Any line items visible on this page

Return the same JSON structure as before, using null for fields not visible on this page."""

    def getSubsequentPagePrompt(self):
        return """This is a continuation page of an invoice document. Focus on extracting:
1. Additional line items or services
2. Any additional charges or discounts
3. Continuation of vendor or billing information
4. Additional notes or terms

Return the same JSON structure, focusing on line_items array and any additional information."""

    def getOcrEnhancedPrompt(self, extractedText, customPrompt=None):
        basePrompt = customPrompt if customPrompt else self.getComprehensiveExtractionPrompt()
        return f"""The following text was extracted via OCR from an invoice document:

{extractedText}

{basePrompt}

Note: This text was OCR-extracted, so there may be some character recognition errors. Use context and common invoice patterns to interpret unclear text."""

    def getEnhancedExtractionPrompt(self, originalData, customPrompt=None):
        missingFields = []
        criticalFields = ['invoice_number', 'vendor_name', 'total_amount', 'invoice_date']
        
        for field in criticalFields:
            if not originalData.get(field):
                missingFields.append(field)
        
        focusText = f"Pay special attention to extracting: {', '.join(missingFields)}" if missingFields else "Focus on improving extraction accuracy"
        
        return f"""{self.getComprehensiveExtractionPrompt()}

ENHANCED EXTRACTION FOCUS:
{focusText}

Look more carefully for:
- Alternative formats for dates (MM/DD/YYYY, DD-MM-YYYY, etc.)
- Invoice numbers in headers, footers, or reference sections
- Vendor names in letterheads or signature blocks
- Amounts in different currency formats or locations
- Bank details in payment instruction sections"""

    def mergePageExtractions(self, allExtractions):
        if not allExtractions:
            return {}
        
        bestExtraction = max(allExtractions, key=lambda x: x['confidence'])
        mergedData = bestExtraction['data'].copy()
        
        allLineItems = []
        for extraction in allExtractions:
            pageItems = extraction['data'].get('line_items', [])
            if pageItems:
                allLineItems.extend(pageItems)
        
        if allLineItems:
            mergedData['line_items'] = allLineItems
        
        for extraction in allExtractions:
            for key, value in extraction['data'].items():
                if key not in mergedData or not mergedData[key]:
                    if value and value != 'null':
                        mergedData[key] = value
        
        logger.info(f"Merged data from {len(allExtractions)} pages")
        return mergedData

    def parseTextractResponse(self, response):
        extractedText = ""
        
        for block in response['Blocks']:
            if block['BlockType'] == 'LINE':
                extractedText += block['Text'] + "\n"
        
        return extractedText

    def processTextWithBedrock(self, text, prompt):
        try:
            requestBody = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4000,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"{prompt}\n\nText to extract from:\n{text}"
                            }
                        ]
                    }
                ]
            }
            
            response = self.bedrockClient.invoke_model(
                modelId='us.anthropic.claude-sonnet-4-20250514-v1:0',
                contentType='application/json',
                accept='application/json',
                body=json.dumps(requestBody)
            )
            
            responseBody = json.loads(response['body'].read())
            extractedContent = responseBody['content'][0]['text']
            cleanedContent = self.cleanJsonResponse(extractedContent)
            
            return json.loads(cleanedContent)
            
        except Exception as e:
            logger.error(f"Text processing with Bedrock failed: {e}")
            raise

    def extractTablesFromPdf(self, pdfBytes):
        try:
            logger.info("Extracting table data using Textract")
            
            response = self.textractClient.analyze_document(
                Document={'Bytes': pdfBytes},
                FeatureTypes=['TABLES']
            )
            
            tables = []
            tableBlocks = {}
            cellBlocks = {}
            
            for block in response['Blocks']:
                if block['BlockType'] == 'TABLE':
                    tableBlocks[block['Id']] = block
                elif block['BlockType'] == 'CELL':
                    cellBlocks[block['Id']] = block
            
            for tableId, table in tableBlocks.items():
                tableData = self.parseTableStructure(table, cellBlocks, response['Blocks'])
                if tableData and len(tableData) > 1:
                    tables.append(tableData)
            
            logger.info(f"Extracted {len(tables)} tables from PDF")
            return tables
            
        except Exception as e:
            logger.error(f"Table extraction failed: {e}")
            return []

    def parseTableStructure(self, table, cellBlocks, allBlocks):
        try:
            rows = {}
            
            if 'Relationships' not in table:
                return []
            
            for relationship in table['Relationships']:
                if relationship['Type'] == 'CHILD':
                    for childId in relationship['Ids']:
                        if childId in cellBlocks:
                            cell = cellBlocks[childId]
                            rowIndex = cell.get('RowIndex', 1) - 1
                            colIndex = cell.get('ColumnIndex', 1) - 1
                            
                            if rowIndex not in rows:
                                rows[rowIndex] = {}
                            
                            cellText = self.getCellText(cell, allBlocks)
                            rows[rowIndex][colIndex] = cellText.strip()
            
            tableData = []
            for rowIndex in sorted(rows.keys()):
                row = rows[rowIndex]
                rowData = []
                maxCol = max(row.keys()) if row else 0
                
                for colIndex in range(maxCol + 1):
                    cellValue = row.get(colIndex, "")
                    rowData.append(cellValue)
                
                tableData.append(rowData)
            
            return tableData
            
        except Exception as e:
            logger.error(f"Table parsing failed: {e}")
            return []

    def getCellText(self, cell, allBlocks):
        text = ""
        
        if 'Relationships' in cell:
            for relationship in cell['Relationships']:
                if relationship['Type'] == 'CHILD':
                    for childId in relationship['Ids']:
                        childBlock = next((block for block in allBlocks if block['Id'] == childId), None)
                        if childBlock and childBlock['BlockType'] == 'WORD':
                            text += childBlock.get('Text', '') + " "
        
        return text.strip()

    def enhanceExtractionWithTables(self, extractedData, pdfBytes):
        try:
            if extractedData.get('line_items'):
                logger.info("Line items already extracted, skipping table enhancement")
                return extractedData
            
            tables = self.extractTablesFromPdf(pdfBytes)
            
            if not tables:
                logger.info("No tables found for enhancement")
                return extractedData
            
            lineItems = self.convertTablesToLineItems(tables)
            
            if lineItems:
                extractedData['line_items'] = lineItems
                logger.info(f"Enhanced extraction with {len(lineItems)} line items from tables")
            
            return extractedData
            
        except Exception as e:
            logger.error(f"Table enhancement failed: {e}")
            return extractedData

    def convertTablesToLineItems(self, tables):
        lineItems = []
        
        for table in tables:
            if len(table) < 2:
                continue
            
            headers = [col.lower().strip() for col in table[0]]
            
            descIndex = self.findColumnIndex(headers, ['description', 'item', 'product', 'service'])
            qtyIndex = self.findColumnIndex(headers, ['qty', 'quantity', 'amount'])
            priceIndex = self.findColumnIndex(headers, ['price', 'unit price', 'rate', 'cost'])
            totalIndex = self.findColumnIndex(headers, ['total', 'amount', 'subtotal'])
            
            for row in table[1:]:
                if len(row) <= max(filter(None, [descIndex, qtyIndex, priceIndex, totalIndex])):
                    continue
                
                lineItem = {}
                
                if descIndex is not None and descIndex < len(row):
                    lineItem['description'] = row[descIndex]
                
                if qtyIndex is not None and qtyIndex < len(row):
                    qty = self.parseNumber(row[qtyIndex])
                    if qty is not None:
                        lineItem['quantity'] = qty
                
                if priceIndex is not None and priceIndex < len(row):
                    price = self.parseNumber(row[priceIndex])
                    if price is not None:
                        lineItem['unit_price'] = price
                
                if totalIndex is not None and totalIndex < len(row):
                    total = self.parseNumber(row[totalIndex])
                    if total is not None:
                        lineItem['total'] = total
                
                if lineItem.get('description') and (lineItem.get('quantity') or lineItem.get('total')):
                    lineItems.append(lineItem)
        
        return lineItems

    def findColumnIndex(self, headers, keywords):
        for i, header in enumerate(headers):
            for keyword in keywords:
                if keyword in header:
                    return i
        return None

    def parseNumber(self, text):
        if not text:
            return None
        
        text = str(text).strip()
        
        text = re.sub(r'[,$]', '', text)
        
        try:
            if '.' in text:
                return float(text)
            else:
                return int(text)
        except ValueError:
            return None

    def processWithBedrock(self, pdfData, customPrompt=None):
        try:
            prompt = customPrompt if customPrompt else self.getComprehensiveExtractionPrompt()
            
            logger.info(f"Using prompt length: {len(prompt)} characters")
            logger.info(f"PDF data length: {len(pdfData)} characters")
            
            requestBody = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4000,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": pdfData
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            }
            
            logger.info("Calling Bedrock Claude Sonnet 4...")
            response = self.bedrockClient.invoke_model(
                modelId='us.anthropic.claude-sonnet-4-20250514-v1:0',
                contentType='application/json',
                accept='application/json',
                body=json.dumps(requestBody)
            )
            
            responseBody = json.loads(response['body'].read())
            extractedContent = responseBody['content'][0]['text']
            
            logger.info(f"Bedrock response received - Length: {len(extractedContent)} characters")
            
            cleanedContent = self.cleanJsonResponse(extractedContent)
            
            try:
                parsedData = json.loads(cleanedContent)
                logger.info("JSON parsing successful")
                return parsedData
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Raw response: {extractedContent[:500]}...")
                raise Exception(f"Invalid JSON response from Bedrock: {str(e)}")
            
        except ClientError as e:
            errorCode = e.response['Error']['Code']
            if errorCode == 'ValidationException':
                logger.error("Bedrock validation error - likely invalid base64 data")
            elif errorCode == 'AccessDeniedException':
                logger.error("Bedrock access denied - check IAM permissions")
            elif errorCode == 'ThrottlingException':
                logger.error("Bedrock throttling - too many requests")
            else:
                logger.error(f"Bedrock error {errorCode}: {e}")
            raise Exception(f"Bedrock processing failed: {str(e)}")
        except Exception as e:
            logger.error(f"Bedrock processing error: {e}")
            raise

    def cleanJsonResponse(self, responseText):
        logger.info("Cleaning JSON response from Bedrock")
        
        if '```json' in responseText:
            logger.info("Removing ```json markdown blocks")
            responseText = responseText.split('```json')[1]
            if '```' in responseText:
                responseText = responseText.split('```')[0]
        elif '```' in responseText:
            logger.info("Removing ``` markdown blocks")
            parts = responseText.split('```')
            if len(parts) >= 3:
                responseText = parts[1]
        
        cleaned = responseText.strip()
        logger.info(f"JSON response cleaned - Length: {len(cleaned)}")
        
        return cleaned

    def calculateExtractionConfidence(self, extractedData):
        if not extractedData:
            return 0.0
        
        criticalFields = ['invoice_number', 'vendor_name', 'total_amount', 'invoice_date']
        importantFields = ['due_date', 'currency', 'vendor_address', 'line_items', 'tax_amount']
        optionalFields = ['vendor_email', 'vendor_phone', 'purchase_order', 'payment_terms']
        
        criticalScore = sum(1 for field in criticalFields if extractedData.get(field) not in [None, '', 'null']) / len(criticalFields)
        importantScore = sum(1 for field in importantFields if extractedData.get(field) not in [None, '', 'null', []]) / len(importantFields)
        optionalScore = sum(1 for field in optionalFields if extractedData.get(field) not in [None, '', 'null']) / len(optionalFields)
        
        baseConfidence = (criticalScore * 0.6) + (importantScore * 0.3) + (optionalScore * 0.1)
        
        qualityBonus = self.calculateDataQualityBonus(extractedData)
        validationPenalty = self.calculateValidationPenalty(extractedData)
        
        finalConfidence = baseConfidence + qualityBonus - validationPenalty
        
        return round(max(0.0, min(1.0, finalConfidence)), 2)

    def calculateDataQualityBonus(self, extractedData):
        bonus = 0.0
        
        if self.isValidDate(extractedData.get('invoice_date')):
            bonus += 0.05
        
        if self.isValidAmount(extractedData.get('total_amount')):
            bonus += 0.05
        
        if self.isValidInvoiceNumber(extractedData.get('invoice_number')):
            bonus += 0.03
        
        lineItems = extractedData.get('line_items', [])
        if isinstance(lineItems, list) and len(lineItems) > 0:
            validLineItems = sum(1 for item in lineItems if self.isValidLineItem(item))
            if validLineItems == len(lineItems):
                bonus += 0.05
        
        if self.hasConsistentMath(extractedData):
            bonus += 0.07
        
        return bonus

    def calculateValidationPenalty(self, extractedData):
        penalty = 0.0
        
        if not self.isValidCurrency(extractedData.get('currency')):
            penalty += 0.02
        
        if self.hasInconsistentAmounts(extractedData):
            penalty += 0.1
        
        if self.hasInvalidDates(extractedData):
            penalty += 0.05
        
        if self.hasEmptyCriticalFields(extractedData):
            penalty += 0.15
        
        return penalty

    def isValidDate(self, dateStr):
        if not dateStr or dateStr == 'null':
            return False
        
        try:
            if re.match(r'^\d{4}-\d{2}-\d{2}$', str(dateStr)):
                datetime.strptime(str(dateStr), '%Y-%m-%d')
                return True
        except ValueError:
            pass
        
        return False

    def isValidAmount(self, amount):
        if amount is None or amount == 'null':
            return False
        
        try:
            float_amount = float(amount)
            return float_amount >= 0
        except (ValueError, TypeError):
            return False

    def isValidInvoiceNumber(self, invNum):
        if not invNum or invNum == 'null':
            return False
        
        invStr = str(invNum).strip()
        return len(invStr) >= 3 and not invStr.lower() in ['unknown', 'n/a', 'none']

    def isValidCurrency(self, currency):
        if not currency or currency == 'null':
            return True
        
        validCurrencies = ['USD', 'EUR', 'GBP', 'CAD', 'AUD', 'JPY', '$', '€', '£']
        return str(currency).upper() in validCurrencies or currency in validCurrencies

    def isValidLineItem(self, item):
        if not isinstance(item, dict):
            return False
        
        hasDescription = bool(item.get('description'))
        hasQuantityOrTotal = bool(item.get('quantity')) or bool(item.get('total'))
        
        return hasDescription and hasQuantityOrTotal

    def hasConsistentMath(self, extractedData):
        try:
            subtotal = extractedData.get('subtotal')
            taxAmount = extractedData.get('tax_amount')
            totalAmount = extractedData.get('total_amount')
            
            if all(x is not None for x in [subtotal, taxAmount, totalAmount]):
                expected = float(subtotal) + float(taxAmount)
                actual = float(totalAmount)
                return abs(expected - actual) < 0.01
            
            lineItems = extractedData.get('line_items', [])
            if lineItems and totalAmount:
                lineTotal = sum(float(item.get('total', 0)) for item in lineItems if item.get('total'))
                if lineTotal > 0:
                    return abs(lineTotal - float(totalAmount)) < (float(totalAmount) * 0.1)
            
            return True
            
        except (ValueError, TypeError):
            return True

    def hasInconsistentAmounts(self, extractedData):
        try:
            amounts = []
            for field in ['total_amount', 'subtotal', 'tax_amount']:
                value = extractedData.get(field)
                if value is not None:
                    amounts.append(float(value))
            
            if len(amounts) >= 2:
                maxAmount = max(amounts)
                return any(amount < 0 for amount in amounts) or maxAmount > 1000000
            
            return False
            
        except (ValueError, TypeError):
            return True

    def hasInvalidDates(self, extractedData):
        dateFields = ['invoice_date', 'due_date']
        
        for field in dateFields:
            dateValue = extractedData.get(field)
            if dateValue and dateValue != 'null':
                if not self.isValidDate(dateValue):
                    return True
        
        try:
            invDate = extractedData.get('invoice_date')
            dueDate = extractedData.get('due_date')
            
            if invDate and dueDate and self.isValidDate(invDate) and self.isValidDate(dueDate):
                invDateObj = datetime.strptime(invDate, '%Y-%m-%d')
                dueDateObj = datetime.strptime(dueDate, '%Y-%m-%d')
                
                if dueDateObj < invDateObj:
                    return True
        
        except ValueError:
            return True
        
        return False

    def hasEmptyCriticalFields(self, extractedData):
        criticalFields = ['invoice_number', 'vendor_name', 'total_amount']
        emptyCount = sum(1 for field in criticalFields if not extractedData.get(field) or extractedData.get(field) == 'null')
        return emptyCount >= 2

    def validateAndCleanData(self, extractedData):
        logger.info("Validating and cleaning extracted data")
        
        cleanedData = extractedData.copy()
        
        cleanedData = self.cleanAmountFields(cleanedData)
        cleanedData = self.cleanDateFields(cleanedData)
        cleanedData = self.cleanLineItems(cleanedData)
        cleanedData = self.removeEmptyFields(cleanedData)
        
        validationResults = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        if self.hasInconsistentAmounts(cleanedData):
            validationResults['warnings'].append('Inconsistent amount calculations detected')
        
        if self.hasInvalidDates(cleanedData):
            validationResults['errors'].append('Invalid date format or logic detected')
            validationResults['is_valid'] = False
        
        if self.hasEmptyCriticalFields(cleanedData):
            validationResults['errors'].append('Missing critical fields (invoice_number, vendor_name, total_amount)')
            validationResults['is_valid'] = False
        
        cleanedData['validation'] = validationResults
        
        logger.info(f"Data validation complete - Valid: {validationResults['is_valid']}, Warnings: {len(validationResults['warnings'])}, Errors: {len(validationResults['errors'])}")
        
        return cleanedData

    def cleanAmountFields(self, data):
        amountFields = ['total_amount', 'subtotal', 'tax_amount']
        
        for field in amountFields:
            value = data.get(field)
            if value is not None and value != 'null':
                try:
                    cleanValue = str(value).replace(',', '').replace('$', '').strip()
                    data[field] = float(cleanValue)
                except (ValueError, TypeError):
                    data[field] = None
        
        return data

    def cleanDateFields(self, data):
        dateFields = ['invoice_date', 'due_date']
        
        for field in dateFields:
            value = data.get(field)
            if value and value != 'null':
                cleanDate = self.normalizeDate(str(value))
                data[field] = cleanDate
        
        return data

    def cleanLineItems(self, data):
        lineItems = data.get('line_items', [])
        
        if isinstance(lineItems, list):
            cleanedItems = []
            for item in lineItems:
                if isinstance(item, dict) and item.get('description'):
                    cleanedItem = item.copy()
                    
                    for amountField in ['quantity', 'unit_price', 'total']:
                        value = cleanedItem.get(amountField)
                        if value is not None:
                            try:
                                cleanedItem[amountField] = float(str(value).replace(',', '').replace('$', ''))
                            except (ValueError, TypeError):
                                cleanedItem[amountField] = None
                    
                    cleanedItems.append(cleanedItem)
            
            data['line_items'] = cleanedItems
        
        return data

    def removeEmptyFields(self, data):
        cleanedData = {}
        
        for key, value in data.items():
            if value is not None and value != 'null' and value != '':
                if isinstance(value, list) and len(value) == 0:
                    continue
                cleanedData[key] = value
        
        return cleanedData

    def normalizeDate(self, dateStr):
        if not dateStr:
            return None
        
        dateStr = str(dateStr).strip()
        
        patterns = [
            ('%Y-%m-%d', r'^\d{4}-\d{2}-\d{2}$'),
            ('%m/%d/%Y', r'^\d{1,2}/\d{1,2}/\d{4}$'),
            ('%d/%m/%Y', r'^\d{1,2}/\d{1,2}/\d{4}$'),
            ('%m-%d-%Y', r'^\d{1,2}-\d{1,2}-\d{4}$'),
            ('%d-%m-%Y', r'^\d{1,2}-\d{1,2}-\d{4}$'),
        ]
        
        for pattern, regex in patterns:
            if re.match(regex, dateStr):
                try:
                    dateObj = datetime.strptime(dateStr, pattern)
                    return dateObj.strftime('%Y-%m-%d')
                except ValueError:
                    continue
        
        return None

    def getErrorSuggestion(self, errorCode):
        suggestions = {
            'ValidationException': 'Check if PDF data is valid base64 and file is not corrupted',
            'AccessDeniedException': 'Verify Lambda has bedrock:InvokeModel permissions',
            'ThrottlingException': 'Implement retry logic or request rate limit increase',
            'NoSuchKey': 'Verify S3 file path and ensure file exists',
            'AccessDenied': 'Check Lambda has s3:GetObject permissions for the bucket'
        }
        return suggestions.get(errorCode, 'Check AWS documentation for this error code')

    def getComprehensiveExtractionPrompt(self):
        return """You are an expert invoice data extraction system. Analyze this invoice document thoroughly and extract ALL relevant information with maximum accuracy.

Return the data as a JSON object with this exact structure. Extract precisely what you see - do not infer, assume, or modify anything:

{
    "invoice_number": "exact invoice/reference number as shown",
    "invoice_date": "date in YYYY-MM-DD format",
    "due_date": "due date in YYYY-MM-DD format if shown",
    "vendor_name": "full vendor/supplier company name",
    "vendor_address": "complete vendor address as shown",
    "vendor_email": "vendor email if present",
    "vendor_phone": "vendor phone if present",
    "vendor_tax_id": "tax ID/VAT number if present",
    "vendor_website": "website if present",
    "bill_to_name": "billing company/person name",
    "bill_to_address": "complete billing address",
    "ship_to_name": "shipping company/person name if different",
    "ship_to_address": "shipping address if present",
    "purchase_order": "PO number if referenced",
    "total_amount": "final total amount as number",
    "subtotal": "subtotal before tax as number",
    "tax_amount": "total tax amount as number",
    "currency": "currency code or symbol as shown",
    "payment_terms": "payment terms description",
    "payment_method": "accepted payment methods if listed",
    "bank_details": {
        "account_name": "bank account name if present",
        "account_number": "account number if present", 
        "routing_number": "routing/sort code if present",
        "bank_name": "bank name if present",
        "swift_code": "SWIFT/BIC code if present"
    },
    "line_items": [
        {
            "description": "item/service description",
            "quantity": "quantity as number",
            "unit_price": "price per unit as number",
            "total": "line total as number",
            "tax_rate": "tax percentage if shown",
            "product_code": "SKU/product code if present"
        }
    ],
    "additional_charges": [
        {
            "description": "charge description (shipping, handling, etc.)",
            "amount": "charge amount as number"
        }
    ],
    "discounts": [
        {
            "description": "discount description",
            "amount": "discount amount as number"
        }
    ],
    "notes": "any additional notes or special instructions",
    "document_type": "type of document (invoice, credit note, etc.)"
}

CRITICAL EXTRACTION RULES:
1. Extract EXACTLY what is written - preserve original spelling and case
2. Use null for any fields that are not present or clearly visible
3. For amounts, extract as numbers without currency symbols (e.g., 1500.50 not "$1,500.50")
4. For dates, convert to YYYY-MM-DD format, use null if format is unclear
5. If multiple currencies appear, note the primary transaction currency
6. Include ALL line items, charges, and discounts that are visible
7. For addresses, include complete text including postal codes
8. For tax IDs, include exactly as shown (with formatting)
9. Return ONLY the JSON object, no explanations or additional text
10. Ensure all amounts are mathematically consistent

Double-check your extraction for accuracy before responding. Begin extraction:"""


class DataProcessor:
    def createDataFrame(self, extractedData):
        invoiceDate = extractedData.get('invoice_date')
        dueDate = extractedData.get('due_date')
        
        bulkPaymentData = {
            'vendorName': extractedData.get('vendor_name') or None,
            'accountNumber': extractedData.get('bank_details', {}).get('account_number') or None,
            'routingNumber': extractedData.get('bank_details', {}).get('routing_number') or None,
            'paymentAmount': extractedData.get('total_amount') or 0,
            'invoiceNumber': extractedData.get('invoice_number') or None,
            'invoiceDate': invoiceDate if invoiceDate and invoiceDate.strip() and invoiceDate != 'null' else None,
            'dueDate': dueDate if dueDate and dueDate.strip() and dueDate != 'null' else None,
            'currency': extractedData.get('currency', 'USD'),
            'paymentReference': f"INV-{extractedData.get('invoice_number', 'UNKNOWN')}",
            'bankName': extractedData.get('bank_details', {}).get('bank_name') or None,
            'swiftCode': extractedData.get('bank_details', {}).get('swift_code') or None,
            'vendorAddress': extractedData.get('vendor_address') or None,
            'processedDate': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return pd.DataFrame([bulkPaymentData])


class ExcelProcessor:
    def __init__(self, s3Client):
        self.s3Client = s3Client
    
    def uploadToS3(self, dataFrame, bucketName):
        excelBuffer = BytesIO()
        with pd.ExcelWriter(excelBuffer, engine='openpyxl') as writer:
            dataFrame.to_excel(writer, sheet_name='BulkPayments', index=False)
        
        excelBuffer.seek(0)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        invoiceNumber = dataFrame.iloc[0]['invoiceNumber'] if not dataFrame.empty else 'UNKNOWN'
        excelKey = f"processed-invoices/{timestamp}_{invoiceNumber}_bulk_payment.xlsx"
        
        self.s3Client.put_object(
            Bucket=bucketName,
            Key=excelKey,
            Body=excelBuffer.getvalue(),
            ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
        logger.info(f"Excel file uploaded to s3://{bucketName}/{excelKey}")
        return excelKey


class DatabaseProcessor:
    def __init__(self):
        self.dbConfig = {
            'host': os.environ.get('DB_HOST'),
            'database': os.environ.get('DB_NAME'),
            'user': os.environ.get('DB_USER'),
            'password': os.environ.get('DB_PASSWORD'),
            'port': os.environ.get('DB_PORT', 5432)
        }
    
    def insertToPostgres(self, dataFrame):
        try:
            connection = psycopg2.connect(**self.dbConfig)
            cursor = connection.cursor()
            
            createTableQuery = """
            CREATE TABLE IF NOT EXISTS bulk_payments (
                id SERIAL PRIMARY KEY,
                vendor_name VARCHAR(255),
                account_number VARCHAR(50),
                routing_number VARCHAR(20),
                payment_amount DECIMAL(12,2),
                invoice_number VARCHAR(100),
                invoice_date DATE,
                due_date DATE,
                currency VARCHAR(10),
                payment_reference VARCHAR(100),
                bank_name VARCHAR(255),
                swift_code VARCHAR(20),
                vendor_address TEXT,
                processed_date TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            cursor.execute(createTableQuery)
            
            for _, row in dataFrame.iterrows():
                insertQuery = """
                INSERT INTO bulk_payments (
                    vendor_name, account_number, routing_number, payment_amount,
                    invoice_number, invoice_date, due_date, currency,
                    payment_reference, bank_name, swift_code, vendor_address, processed_date
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(insertQuery, (
                    row['vendorName'], row['accountNumber'], row['routingNumber'],
                    row['paymentAmount'], row['invoiceNumber'], row['invoiceDate'],
                    row['dueDate'], row['currency'], row['paymentReference'],
                    row['bankName'], row['swiftCode'], row['vendorAddress'],
                    row['processedDate']
                ))
            
            connection.commit()
            logger.info(f"Successfully inserted {len(dataFrame)} records into PostgreSQL")
            
        except Exception as e:
            logger.error(f"Database error: {e}")
            raise
        finally:
            if connection:
                cursor.close()
                connection.close()


def handler(event, context):
    processor = InvoiceProcessor()
    return processor.lambdaHandler(event, context)

