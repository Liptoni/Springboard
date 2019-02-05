# Submission

Name: Ian Lipton

Date: 2/4/2019

## Instructions

I created a SQL relational database to house the data. The database has the following tables:
*	Import_Log – This holds the filepaths of files imported into the database. Used to ensure that we don’t import the same data twice
*	Trees – contains information about trees, joins with Apples and Bananas on Tree_ID
*	Apples – info about apples
*	Bananas  - info about bananas
*	Cows  - info about individual cows joins with milk on Cow_ID
*	Milk – info about individual milk cartons joins with Butter on Cow_ID and Carton_Number
*	Butter – info about individual sticks of butter
*	Sales – information about sold items including sale date

![Database Diagram](./DB_Diagram.PNG?raw=True "Database Diagram")

I wrote a python compiler script to loop over the folders and files in the files directory and generate tables to be imported into the SQL database. For this exercise, I imported these files by hand. In the future, this would be unrealistically tedious and I would write a script that would call my compiler script and then insert the new data into the database. This could be done on any time frame (daily, weekly, etc.). Here is the script I used to compile all of the data files:

```python
import os
import re
import pandas as pd
import PyPDF2
import datetime

#get date that this script was run (used to date output excel files)
this_date = datetime.datetime.today().strftime('%Y_%m_%d')


def get_age(age):
    """
    Takes a string containing an age
    Returns an integer age
    """
    #find string with 0 or more non-digits followed by any number of digits
    source_age = re.search('\D*(\d+)', age)
    return int(source_age.group(1))

#create a dictionary with farm types for different farm directories
farm_types = {'Bananarama': 'bananas', 'Heritage Farms': 'apples', 'Old McDonald': 'milk'}
order_dirs = next(os.walk('files/orders'))[1]


#instantiate empty lists that will hold dataframes to concatenate
imported_files = []
banana_dfs = []
tree_dfs = []
apple_dfs = []
milk_dfs = []
cow_dfs = []
cow_measures = []

#loop over order directories
for direct in order_dirs:
    farm_type = farm_types[direct]
    directory_path = 'files/orders/'+direct
    #get a list of data files to read
    data_files = []
    for path, subdirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.pdf') or file.endswith('.csv'):
                data_files.append(os.path.join(path, file))            
    #loop over data files
    for data_file in data_files:
        #append each file to the imported files list
        imported_files.append(data_file)
        
        #check farm types and then parse data accordingly
        if farm_type == 'bananas':
            #read file
            df = pd.read_csv(data_file)
            
            #get banana specific data
            banana_df = df[['ID', 'Tree ID', 'Generation number', 'Expiration date']]
            banana_df.columns = ['Banana_ID', 'Tree_ID', 'Generation_Number', 'Expiration_Date']
            banana_dfs.append(banana_df)            
            
            #get tree specific data
            tree_df = df[['Tree ID', 'Product', 'Variety']]
            tree_df['age'] = -999
            tree_df['farm'] = direct
            tree_df.columns = ['Tree_ID', 'Tree_Type', 'Tree_Variety', 'Tree_Age_yrs', 'Farm']
            tree_dfs.append(tree_df)
    
        if farm_type == 'apples':
            #read file
            df = pd.read_csv(data_file)
            
            #get apple specific data
            apple_df = df[['ID', 'Tree ID', 'Color', 'Pollination group', 'Expiration date']]
            apple_df.columns = ['Apple_ID', 'Tree_ID', 'Color', 'Pollination_Group', 'Expiration_Date']
            apple_dfs.append(apple_df)
            
            #get tree specific data
            tree_df = df[['Tree ID', 'Product', 'Variety']]
            tree_df['age'] = df['Age of tree'].apply(get_age)
            tree_df['farm'] = direct 
            tree_df.columns = ['Tree_ID', 'Tree_Type', 'Tree_Variety', 'Tree_Age_yrs', 'Farm']
            tree_dfs.append(tree_df)
        
        
        if farm_type == 'milk':            
            if data_file.endswith('.csv'):
                df = pd.read_csv(data_file)                
                #get cow data
                cow_df = df[['Cow ID', 'Cow breed']]                
                cow_df['Cow_Age'] = df['Cow age'].apply(get_age)
                cow_df['Farm'] = direct
                cow_df.columns = ['Cow_ID', 'Cow_Breed', 'Cow_Age', 'Farm']
                cow_dfs.append(cow_df)
                
                #get milk data
                milk_cartons = []
                #loop over each row and create a record for each individual carton
                for intex, row in df.iterrows():
                    cow_id = row['Cow ID']
                    expir = row['Expiration date']
                    vendor = row['Vendor']
                    cartons= int(row['# cartons (milk)'])
                    for carton in range(cartons):
                        carton_number = carton + 1
                        milk_cartons.append([cow_id, carton_number, expir, vendor])
                
                #concatenate milk carton lists into dataframe
                milk_df_headers = ['Cow_ID','Carton_Number', 'Expiration_Date', 'Vendor']
                milk_df = pd.DataFrame(milk_cartons, columns = milk_df_headers)
                milk_dfs.append(milk_df)
            
            #these pdfs have length and weight values for the cows
            if data_file.endswith('.pdf'):                
                #get text from the PDFs
                with open (data_file, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfFileReader(pdf_file)
                    page = pdf_reader.getPage(0)
                    text = page.extractText()
                    
                    #parse the text from the PDF to get the values
                    find_nums = re.search('\D*(\d+)\D*(\d+)\D*(\d+)', text)
                    cow_id = int(find_nums.group(1))
                    length = int(find_nums.group(2))
                    weight = int(find_nums.group(3))
                    
                    #will use these later to create a df and join with cow_dfs
                    cow_measures.append([cow_id, length, weight])

#concatenate dataframes and then export to excel. These excel files will be imported into the relational database           
cow_df = pd.concat(cow_dfs)
cow_meas_df = pd.DataFrame(cow_measures, columns = ['Cow_ID', 'Cow_Length_ft', 'Cow_Weight_lbs'])
cow_df = cow_df.merge(cow_meas_df, how='left', left_on='Cow_ID', right_on='Cow_ID')
cow_df = cow_df[['Cow_ID', 'Cow_Age', 'Cow_Breed', 'Cow_Length_ft', 'Cow_Weight_lbs', 'Farm']]
cow_file_name = this_date + "_cows.xlsx"
cow_df.to_excel(cow_file_name, index=False)

banana_df = pd.concat(banana_dfs)
banana_file_name = this_date + "_bananas.xlsx"
banana_df.to_excel(banana_file_name, index=False)

tree_df = pd.concat(tree_dfs)
tree_file_name = this_date + "_trees.xlsx"
tree_df.to_excel(tree_file_name, index=False)

apple_df = pd.concat(apple_dfs)
apple_file_name = this_date + "_apples.xlsx"
apple_df.to_excel(apple_file_name, index=False)

milk_df = pd.concat(milk_dfs)
milk_file_name = this_date + "_milk.xlsx"
milk_df.to_excel(milk_file_name, index=False)

#loop through Spring Foods directories          
spring_dirs = next(os.walk('files/Spring Foods'))[1]
for direct in spring_dirs:
    if direct == 'Butter logs':
        directory_path = 'files/Spring Foods/Butter logs'
        data_files = []
        for path, subdirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.csv'):
                    data_files.append(os.path.join(path, file))
        
        butter_info = []
        butter_id = 1 #future imports use max butter_id from sql db
        for data_file in data_files:
            imported_files.append(data_file)
            #get the date that the butter was made from the name of the directory
            made_date = re.findall(r'\d\d\d\d-\d\d-\d\d', data_file)
            df = pd.read_csv(data_file)            
            #loop over each row to get the number of sticks and identifying info
            for index, row in df.iterrows():
                carton_info = re.search('\D*(\d+)\D*(\d+)', row['Carton of milk'])
                cow_id = int(carton_info.group(1))
                carton = int(carton_info.group(2))
                churner = row['Churner']
                exp_date = row['Expiration date']
                sticks = int(row['# sticks of butter'])
                for stick in range(sticks):                    
                    butter_info.append([butter_id, cow_id, carton, churner, made_date[0], exp_date])
                    butter_id += 1
        #concatenate butter dataframes and export to excel
        butter_headers = ['Butter_ID', 'Cow_ID', 'Carton_Number', 'Churner', 'Made_Date', 'Expiration_Date']
        butter_df = pd.DataFrame(butter_info, columns =butter_headers)
        butter_file_name = this_date + "_butter.xlsx"
        butter_df.to_excel(butter_file_name, index=False)
        
    if direct == 'Receipts':
        directory_path = 'files/Spring Foods/Receipts'
        data_files = []
        sales_dfs = []
        for path, subdirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.csv'):
                    data_files.append(os.path.join(path, file))
                    
        sale_id = 1 #future imports use max sale id from sql db
        for data_file in data_files:
            imported_files.append(data_file)
            #get the date of the sale from the directory file path
            sale_date = re.findall(r'\d\d\d\d-\d\d-\d\d', data_file)
            df = pd.read_csv(data_file)
            #want just a float so replace the $ with nothing and cast as float
            df['Price'] = df['Price'].apply(lambda x: float(x.replace('$', '')))
            df['Sale_Date'] = sale_date[0]
            sales_dfs.append(df)
        
        #concatenate sales dataframes and export
        sales_df = pd.concat(sales_dfs)
        sales_df.columns = ['Price', 'Customer', 'ItemID', 'Sale_Date']
        sales_df = sales_df[['Sale_Date', 'Price', 'Customer', 'ItemID']]
        sales_file_name = this_date + "_sales.xlsx"
        sales_df.to_excel(sales_file_name, index=False)
        
#concatenate the imported files list into a dataframe and export
#this will be used in future scripts to ensure that the same files are not imported multiple times
imported_files_df = pd.DataFrame(imported_files, columns = ['file_path'])
imported_file_name = this_date + "_imported_files.xlsx"
imported_files_df.to_excel(imported_file_name, index=False)
```

I ran the following SQL update queries after import to update the consumed and sold fields in the relevant tables:

```sql
update a
set a.Sold = 1, a.Sold_Date = s.Sale_Date
from Apples a
inner join Sales s on a.Apple_ID = s.ItemID

update b
set b.Sold = 1, b.Sold_Date = s.Sale_Date
from Bananas b 
inner join Sales s on b.Banana_ID = s.ItemID

update m
set m.Consumed = 1, m.Consumed_Date = b.Made_Date
from Milk m
inner join Butter b on b.Cow_ID = m.Cow_ID and b.Carton_Number = m.Carton_Number
```

I came across two text files that were not human readable. I attempted, unsuccessfully, to guess the encoding used to generate the files and then determined that they were erroneous text files and chose to move on.


## Questions

1) Why did you settle on this design? Justify any significant design decisions you made (tools you chose, etc.).

I settled on this framework because I think it is the easiest way to keep track of a large number of related items. Being able to track individual items from purchase to consumption, sale, or expiration is easy if all we have to do is track its ID. This framework also allows us to add new types of data on the fly (see my answer to question 6 below) by creating new tables and tracking IDs. We don't have to go back and modify existing records, we can just create new tables and create new metadata for an item by inserting its ID. This framework also works well with various applications. We can create an API, or use existing software that does this for us, to link this backend to some front-end framework which will allow us to give access to users who don't have the ability to write code, or to users who need to enter new data.

The biggest thing I would change right now would be to update how we are storing the IDs for each of our items. Right now they are just integers, but it would be nice to have something else to identify the item type from the ID. This would be necessary if we want to start compiling multiple item types in the same table.

2) What assumptions (if any) did you have to make about the data?

I made one major assumption regarding the sales data. The dates on the directories were both for December of 2019. I assumed that this was erroneous and corrected the dates to 2018. This will affect the count of products available in question three.

Also for the sales data, I assumed that the Item ID referred to either the apples or bananas. But, there was no identifying information about what the product actually was. In the future I would design the sales data to include the item type as well as the ID in case of overlapping IDs from multiple item types. 

I noticed that there were duplicate IDs for bananas from two different trees, Going forward, if we expect this to occur frequently, I would create a composite ID that concatenated both the banana ID and tree ID.

3) How many items did Spring Foods™ have in-stock as of January 28, 2019? (How’d you arrive at this number?)

221. 
I arrived at this answer by running the SQL query posted in the answer to the next question filtering out consumed, sold, and expired items from apples, bananas, milk, and butter:

4) How would technical folks go about answering Question #3? Non-technical folks?
```sql
declare @date_in_question as date
set @date_in_question = '2019-01-28'

select Milk_ID as 'Item_ID', 'milk' as 'item'
from Milk
where Expiration_Date > @date_in_question
	and ((Consumed = 0 and Sold = 0)
		or (Consumed = 1 and Consumed_Date > @date_in_question)
		or (Sold = 1 and Sold_Date > @date_in_question))
union
select Apple_ID as 'Item_ID', 'apple' as 'item'
from Apples
where Expiration_Date > @date_in_question
		and ((Consumed = 0 and Sold = 0)
		or (Consumed = 1 and Consumed_Date > @date_in_question)
		or (Sold = 1 and Sold_Date > @date_in_question))
union
select Ban_UID as 'Item_ID', 'banana' as 'item'
from Bananas
where Expiration_Date > @date_in_question
		and ((Consumed = 0 and Sold = 0)
		or (Consumed = 1 and Consumed_Date > @date_in_question)
		or (Sold = 1 and Sold_Date > @date_in_question))
union
select Butter_ID as 'Item_ID', 'butter' as 'item'
from Butter
where Expiration_Date > @date_in_question
		and Made_Date < @date_in_question
		and ((Consumed = 0 and Sold = 0)
		or (Consumed = 1 and Consumed_Date > @date_in_question)
		or (Sold = 1 and Sold_Date > @date_in_question))
```

In the future, I would probably script something during import to compile a list of all products and their status (consumed, sold, expired, etc.) This would allow someone to query just one table instead of having to write union after union. WIth only four table to query it isn't that bad, but if we start selling more products, this could get out of hand quickly.

For non-technical folks, I envision a front-end application where they can access the data through point and click. For example, we could use Microsoft Access which has the ability to make an ODBC connection to our SQL database and then we can use their built-in form creating tools to create a series of forms that any user could use regardless of technical prowess. I would be able to write some VBA functions that would power these forms and create some stock queries that a user could access. One of those queries could run this question and all we would need from the user is an input in the form of a date.


5) How would you envision future data getting into the system? What processes and technical solutions would need to be put in place? (Imagine that this system is being used by data scientists, cashiers, and others.)

I envision future data getting into the database in two ways. For internally produced items or items that we sell, I envision an application where our staff can indicate items that were consumed or sold. This would then trigger stored procedures that would update the back-end database accordingly. 

For new purchases, I envision a modification of the python compiler script I wrote. This could be called as a command line script, or I could write code to automatically call that compiler script, then trigger the database to import the newly compiled data. My first thought would be to do that by writing VBA function to read in new files, parse them, and then push them to the back-end database.


6) If we were to start freezing some of our items on arrival, what work would be required to track freeze and thaw dates?

The first thing we would need is a new table in our database. This would include the ID of the item being sold, the type of item, the date it was frozen, the date it was thawed, and a new expiration date (assuming the expiration date changes when the item is frozen and thawed). We could join this table back to the individual item tables based on item type and ID. Or, if we have created a single table containing a unique item id, and information about those items, we could join this freeze/thaw table to that by using the unique item ID. We could get this information into the database the same way we get the consumed and sold data in.
