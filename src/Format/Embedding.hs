{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}


-- | The module responsible for parsing embedding vector representations.
--
-- The embedding file consists of a blank-line-separated list of sentences.
-- Each sentence consists of a sequence of lines, and each line is a
-- tab-separated specifiction of a word.  In a line, the first value is the
-- orthographic form, and the remaining @d@ values represent the embedding
-- vector.


module Format.Embedding
  ( readEmbeddings
  ) where


import           GHC.TypeNats (KnownNat)

import qualified Data.Text.Lazy as L
import qualified Data.Text.Lazy.IO as L

import qualified Numeric.LinearAlgebra.Static as LA


-----------------------------------
-- Loading
-----------------------------------


-- | Read the embeddings file.
readEmbeddings
  :: (KnownNat d)
  => FilePath -- ^ Embedding file, one vector per .cupt word
  -> IO [Maybe [LA.R d]]
readEmbeddings = fmap parseEmbeddings . L.readFile


-- | Parse the embeddings.  Returns `Nothing` if one of the words in a
-- sentence has no corresponding vector embedding.
parseEmbeddings
  :: (KnownNat d)
  => L.Text -- ^ Embedding file, one vector per .cupt word
  -> [Maybe [LA.R d]]
parseEmbeddings
  = map parseSent
  . filter (not . L.null)
  . L.splitOn "\n\n"


parseSent :: (KnownNat d) => L.Text -> Maybe [LA.R d]
parseSent
  = sequence
  . map parseToken
  . L.lines


parseToken :: (KnownNat d) => L.Text -> Maybe (LA.R d)
parseToken line =
  let _word : vs0 = L.words line
      vs = map (read . L.unpack) vs0
   in case vs of
        [] -> Nothing
        _  -> Just (LA.fromList vs)
