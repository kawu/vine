{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE DataKinds #-}


module Embedding.Dict
  ( Dict
  , empty
  , load
  , loadSel
  ) where


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


-- | A version of `loadSel` which actually loads the entire dictionary.
load :: (KnownNat n) => Bool -> FilePath -> IO (Dict n)
load = loadSel $ const True


-- | Load a part of the dictionary to memory.  The data format is the same as
-- the one in the pre-trained FastText word embedding files (but the function
-- assumes that no header line is present).
loadSel
  :: (KnownNat n)
  => (T.Text -> Bool) -- ^ Selection function
  -> Bool             -- ^ Has header
  -> FilePath
  -> IO (Dict n)
loadSel select hasHeader path = do
  xs <- L.lines <$> L.readFile path
  let handleHeader = if hasHeader then tail else id
  foldM update empty (handleHeader xs)
  where
    update dict line = do
      let w : vs0 = L.words line
          vs = map (read . L.unpack) vs0
          embedding = LA.fromList vs
      -- print (M.size dict)
      M.size dict `seq`
        if select (L.toStrict w)
           then embedding `seq` return $
             M.insert (L.toStrict w) embedding dict
           else return dict


------------------------------------
-- Testing
------------------------------------


-- main :: IO ()
-- main = do
--   let path = "/datadisk/workdata/fasttext/pre-trained/english/wiki-news-300d-1M-subword.vec"
--   d <- load path :: IO (Dict 300)
--   print $ M.size d
