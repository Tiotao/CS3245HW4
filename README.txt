This is the README file for A0099314Y's and A0099332Y's submission

== General notes about this assignment ==

The strategy for this submission is very similar to HW3, as it is based on tf-idf model (lnc.ltc).
Tokens are stemmed and stopwords eliminated.

Indexing:  The strategy for indexing is very similar to HW3, but instead of treating each document
as a unstructured data, we focus on the Title and Abstract zones. Each zone is indexed separately,
so in the postings file, we may store the term frequency of a word apple in document A00001.xml as
{A00001: {title: 1, desc: 9}} which means that the token "apple" appears 1 time in the title and 9
times in the abstract.

We then calculate the document lengths (by calculating L2 norm), similar to HW3, as we needed it for
vector normalisation when calculating cosine similarities in search phase.

Searching:

We treat query input as two queries: the title query and the description query.

For all terms appearing in a query, we do an OR-ing on all postings list fetched. We then calculates
the cosine similarity between the query and all documents in the postings lists, using lnc.ltc
ranking scheme. In particular, the cosine normalisation is done with both query vector length and
document vector length. The latter is pre-computed during indexing phase, as mentioned earlier.

Since we have zones in this assignment, we have a modification that the two major zones in the files
(title and description) receive zone weighting, which will be applied to their weighted scores. For
example, the title zone may have a weighting of 0.8 while the description has a weighting of 0.2.

After the cosine similarities for title query and description query have been computed, we multiply
the respecitve weighting to the scores, so the weighted sum becomes the score for this particular
document.

Possible enhancements include query expansion. Once we've obtained the preliminary results, we take
the top 10% most relevant documents and concatinate them to form a new query. Then process the new
query in the same algorithm as the initial round, and output the result as the final list of
relevant documents.

In this team project, A0099332Y is in charge of indexing and A0099314Y for searching implementation.

== Files included with this submission ==

.
|-- README.txt ............ this file
|-- dictionary.txt ........ dictonary for all documents
|-- index.py .............. python script to index all documents
|-- postings.txt .......... postings list and document length file
`-- search.py ............. python script to perform query searching
 

== Statement of individual work ==

Please initial one of the following statements.

[x] I, A0099314Y, certify that I have followed the CS 3245 Information
Retrieval class guidelines for homework assignments.  In particular, I
expressly vow that I have followed the Facebook rule in discussing
with others in doing the assignment and did not take notes (digital or
printed) from the discussions.  

[x] I, A0099332Y, certify that I have followed the CS 3245 Information
Retrieval class guidelines for homework assignments.  In particular, I
expressly vow that I have followed the Facebook rule in discussing
with others in doing the assignment and did not take notes (digital or
printed) from the discussions.  

== References ==
