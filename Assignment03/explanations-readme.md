**A1 — Data acquisition and cleaning**

From the data type information above, we can immediately see that there is a mismatch in the data types for the datatime columns `issueddate`, `expireddate` and `extractdate`. We'll fix this by removing the timezone info from `issuedate` and we can drop `extractdate` as that contains data about when the row was filled in. We do not need this for our analysis and clustering with this data.

We can also see that the `numberofemployees` is also float, which should not be possible as employees cannot be in decimals. We checked if any values in the `numberofemployees` column are decimals and they are not, so the data is correct here.

We can see that the values in the geom column does not exist for the entire dataset, so this column can be dropped entirely. Nearly only half of the records have values for `geo_point_2d` and in turn, `geometry`.

As we are going to be plotting these businesses on a map, any business without valid geometry values, longitude, latitude, will be dropped as they cannot be plotted on the map without these values. So let's do that first.

Based on the above value counts and categories for status, we should only keep the business that are `Issued` / active for our EDA, and filter out all that are inactive, pending, gone out of business or cancelled. It's also mentioned on the open data portal for this dataset that: All existing business licences are transitioned automatically into the new categories. Some business licence categories retained the same name. Business owners will renew their licence at the end of each year and may select a different category from the one they were sorted into. New businesses applying for a licence af​ter the update will also select from the updated categories.

As this hints at possible duplicate entries for businesses, it makes sense to avoid these duplicates and keep only the businesses with licences that are curent, active and `Issued`.

We've significantly shrunk the dataset this way according to our needs, ie: 
1. Businesses with valid geometry to pin on a map. 
2. Businesses that are active / have issued licences.
We have also possibly dropped rows with data errors as well. Let's check what the missingness looks like now.

So now, most of the columns do not have missing values. However, `businesssubtype` has 80.6% missing values, `unittype/unit` has ~56%, `businesstradename` has 50.0% and `feepaid` has 30.8%, with >= 10% for `country`, `postalcode`, `expireddate`, `issueddate`, and `businessname`.

We can combine `businesssubtype` and `businesstype` and refine the values to deal with the missing values.

As `businessname` only has three missing values, we can keep it as it is. However, we can drop the businesstradename column due to the large amount of missing values it has, and its similarity with `businessname`.

As we have also filtered this dataset to contain only `Issued` licences, the `status` column becomes redundant as all values in it are now `Issued` and it does not provide additional information now.

As we are plotting this data using the latitude, and longitude values, do not need `geo_point_2d` and `geometry` as we have extracted the latitude and longitude from them.

We know that the `city`, `province` and `country` column only has one unique value, so these columns can be dropped as well, as we are aware that all of these business are located in "Vancouver, BC, CA". These columns do not provide additional information. This data is retained by the `longitude` and `latitude` values when pinned on the map anyway. We can also drop `unit` and `unit type` as address level data is not truly needed for our purpose when we have lat/long available.

As for the rest, like `feepaid` and `numberofemployees` we cannot assume their values to be the `mean` or `median` of the data, as when these are plotted/clustered on the map and can give inaccurate results. We cannot assume values for issued and expired dates as well. Hence, we'll keep the following columns as they are with their missingness for now: feepaid, postalcode, expireddate, issueddate.

Let's consolidate businesstype and businesssubtype into one column. We can drop the `businesssubtype` column then, as we will not need it.

The open data portal mentions that `licencersn` might have duplicate values, let's check and remove any that exist. Hence, let's drop record 52908, as it has missing `unit` data that the other record contains. The only duplicate for `licencersn` has been dropped.

Looking at some of the code results, tables and data descriptions above, we can see that some cols have extremely large amounts of missing values. And some discrepencies exist when you look at the number of unique values for each column. For example: `folderyear` has 3 unique values -> the data from a total of 3 years has been collected. Some duplicate values also exist for `licencenumber` due to revisions, so they can remain in the dataset as they are, since all of these duplicates have unique `licencersn` identifiers. 90 unique `businesstype`s exist and 76 unique `businesssubtype`s exist. We also have some postal code formatting issues where `postalcode` is formatted without a space after the first 3 letter like: `V5T2N4`, instead of `V5T 2N4`.

Fixed postal code formatting.
Consolidated business types with business sub type, and then only kept the categories that refer to 80-90% of the businesses. Hence, kept the top 25 categories that amount to nearly 84% of the business types and grouped the rest as others.

Hence, we can see that the top 25 categories remain that encompass nearly 84% of businesses, while the rest have been grouped as "Other" with nearly 14328 businesses.

---

**A2 — Location-only clustering: K-means vs. DBSCAN**

Picking number of clusters as 4 as after 4, the slope decreases at an almost constant rate.

I chose the DBSCAN values through trail and error, and printing the value counts to ensure noise (-1) is as minimal compared to the actual clustered values.

Based on the k-means clusters, we can see that the 4 clusters divide up into west (red), north (blue), east (pink), and south (turquoise). Mapping this onto the actual map of vancouver (I looked it up on google maps), it seems like the north (blue) part of the cluster shows businesses based mostly in the Stanley Park / Granville Island area, and we can even see the False Creek area in this blue cluster where there are no business due to the water.

The west (red) cluster shows businesses in the West Point Grey and surrounding areas, where there are mostly forests and parks, hence business densities are sparse here, we can see on the plot as well.

The south (turquoise) part of the graph mostly shows the south vancouver area, with businesses mostly clustered around the Fraser River bay area.

The east (pink) part of the graph shows the Downtown eastside area where business are most clustered, and the Kingsway road which was business clustered around its sides, with some divergences into businessed based in Killarney, and Victoria-FraserView, and less density of clusters around the Everett Crowley Park area.

The DBSCAN plot shows one giant orange cluster that dominates over almost every other points / clusters. It also shows a handful of small clusters such as the purple cluster on the far west edge, a couple green dots, a small red cluster bottom right and scattered noise points.

It seems K-means is creating equal sized clusters on the dataset, regaardless of distribution density, while DBSCAN is taking that density into consideration, and it shows us that the big orange cluster is businesses grouped together closely with equal density. The noise points show us where density varies. I think the orange cluster could represent all businesses tightly grouped toegther in downtown while all the other colored points (noise) are those that are physically separated from downtown, either by land, or bodies of water. The noise points are too isolated to belong to the same density cluster.

---

**A3 — Feature-based clustering: Size, Industry, Lifecycle**

So we still have some values missing for feepaid, we can fill them using the mean for this data, as we are assuming we do not have outliers in the data.

PC1 captures the industry category values, while PC2 captures the continuous numerical values we have. The clustered peaks in this graph are likely the encoded industry categories we engineered, and data is grouped across them. The sizes / lengths of these grouped peaks can be explained by the fee paid / number of employee / duration columns.

PC1 explains 29.7% of the data and PC2 explains 17% of the data. Combined, they explain nearly 47.5% of the data variations.

The colored clusters here do not align with the k-means clusters. It seems like k-mean might not be clustering on the bases of industry / business type, but this PCA shows the clustering across industry types and durations / sizes. The points above 20 on the y-axis can be outliers with a very large number of employees, or fee paid, or durations, while most points are concentrated around the y = 0 line (which is the mean), and then, per business type, they are rising up from there.

---

**B4 — from `assignment03B.py`**

Given what I know about the areas in vancouver, we can see that the most dense area containing businesses is Downtown Vancouver with the biggest cluster, with businesses like legal, health, business and retail. Other clusters like Cluster 0: Fairview, Kerrisdale, Kitsilano, Renfrew-Collingwood, Riley Park, West End are grouped together based on their business types: health, rental, retail and restaurants. Hence, all clusters have been grouped together here based on the likeliness of their business type, while cluster sizes depends on how dense / the count of the businesses are in those areas grouped by type.
