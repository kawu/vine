{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE DataKinds #-}


module Embedding.Dict where


-- | Simple embedding dictionary implementation.


import           GHC.TypeNats (KnownNat)
import qualified GHC.TypeNats as Nats

import           Control.Monad (foldM)

import qualified Numeric.LinearAlgebra.Static as LA
import           Numeric.LinearAlgebra.Static (R)
import qualified Data.Text as T
import qualified Data.Text.Lazy as L
import qualified Data.Text.Lazy.IO as L
import qualified Data.Map.Strict as M


------------------------------------
-- Data structures
------------------------------------


-- | Embedding dictionary
type Dict n = M.Map T.Text (R n)


-- | Empty dictionary
empty :: KnownNat n => Dict n
empty = M.empty


------------------------------------
-- Loading
------------------------------------


-- | Load the entire dictionary to memory.  The data format is the same as the
-- one in the pre-trained FastText word embedding files.
load :: (KnownNat n) => FilePath -> IO (Dict n)
load path = do
  xs <- tail . L.lines <$> L.readFile path
  foldM update empty xs
  where
    update dict line = do
      print $ M.size dict
      let w : vs0 = L.words line
          vs = map (read . L.unpack) vs0
          embedding = LA.fromList vs
      return $
        embedding `seq` M.size dict `seq`
        M.insert (L.toStrict w) embedding dict


------------------------------------
-- Testing
------------------------------------


main :: IO ()
main = do
  let path = "/datadisk/workdata/fasttext/pre-trained/english/wiki-news-300d-1M-subword.vec"
  d <- load path :: IO (Dict 300)
  print $ M.size d
