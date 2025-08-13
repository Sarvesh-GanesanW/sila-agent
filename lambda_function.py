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

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class InvoiceProcessor:
    def __init__(self):
        self.s3Client = boto3.client('s3')
        self.bedrockClient = boto3.client('bedrock-runtime', region_name='us-east-1')
        
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
            
            logger.info("Sending PDF to Bedrock for processing")
            extractedData = self.processWithBedrock(pdfData, customPrompt)
            
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
        
        confidence = (criticalScore * 0.6) + (importantScore * 0.3) + (optionalScore * 0.1)
        
        return round(confidence, 2)

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
        bulkPaymentData = {
            'vendorName': extractedData.get('vendor_name', ''),
            'accountNumber': extractedData.get('bank_details', {}).get('account_number', ''),
            'routingNumber': extractedData.get('bank_details', {}).get('routing_number', ''),
            'paymentAmount': extractedData.get('total_amount', 0),
            'invoiceNumber': extractedData.get('invoice_number', ''),
            'invoiceDate': extractedData.get('invoice_date', ''),
            'dueDate': extractedData.get('due_date', ''),
            'currency': extractedData.get('currency', 'USD'),
            'paymentReference': f"INV-{extractedData.get('invoice_number', 'UNKNOWN')}",
            'bankName': extractedData.get('bank_details', {}).get('bank_name', ''),
            'swiftCode': extractedData.get('bank_details', {}).get('swift_code', ''),
            'vendorAddress': extractedData.get('vendor_address', ''),
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


def lambda_handler(event, context):
    processor = InvoiceProcessor()
    return processor.lambdaHandler(event, context)