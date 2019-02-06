{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE TypeOperators #-}


{- Experiments with a DAG-structured recurrent network -}


module Net.DAG where


import           Control.Monad (guard)

import           GHC.Generics (Generic)

import           Lens.Micro.TH (makeLenses)
import           Lens.Micro ((^.))

import qualified Numeric.Backprop as BP
import qualified Numeric.LinearAlgebra.Static.Backprop as LBP
import           Numeric.LinearAlgebra.Static.Backprop
  (R, L, BVar, Reifies, W, (#))

import qualified Data.Map.Strict as M
import qualified Data.DAG as DAG

import           Net.Basic
import qualified Net.FeedForward as FFN
import           Net.FeedForward (FFN(..))


----------------------------------------------
-- Recurrent DAG Network
----------------------------------------------


-- | Encoder RNN
data RecDAG = RecDAG
  { _ffA :: FFN
      10  -- Size of the past state + size of the input word embeddings
      5   -- FFN's internal hidden state
      5   -- Size of the hidden state
    -- ^ To calculate the hidden state of the current arc given the past and
    -- the vector representation of the current arc
  , _ffB :: FFN
      10  -- Size of the hidden sate + size of the input word embeddings
      5   -- FFN's internal hidden state
      1   -- Output: single value
    -- ^ To calculate the probability of the given arc based on the its hidden
    -- representation and its vector representation
  -- TODO: !!!attention calculation!!!
  , _p0  :: R 5
    -- ^ The initial past state
  } deriving (Generic)

instance BP.Backprop RecDAG
makeLenses ''RecDAG


-- | Create a new, random network
new :: IO RecDAG
new =
  RecDAG <$> FFN.new 10 5 5 <*> FFN.new 10 5 1 <*> vector 5


-- | DAG with labeled arcs
type DAG a = DAG.DAG () a


-- | Run the network over a DAG labeled with input word embeddings.
run
  :: BVar s RecDAG
    -- ^ Recurrent DAG network
  -> DAG (BVar s (R 5))
    -- ^ Input DAG with word embeddings
  -> M.Map DAG.EdgeID (BVar s Double)
  -- -> DAG (BVar s Double)
    -- ^ Output DAG with probabilities
run net dag =
  go M.empty (topoEdges dag)
  where
    go hiddenMap [] = hiddenMap
    go hiddenMap (arcID : rest) =
      let arcHidden = calculateHiddenState hiddenMap arcID
       in go (M.insert arcID arcHidden hiddenMap) rest
    calculateHiddenState hiddenMap arcID = undefined


----------------------------------------------
-- DAG Utils
----------------------------------------------


-- | Return the list of the IDs of the DAG arcs sorted topologically
-- (starting with the "root" arcs, with no preceding arcs).
topoEdges :: DAG a -> [DAG.EdgeID]
topoEdges dag =
  case DAG.topoSort dag of
    Nothing -> error "Net.DAG.topoEdges: input graph has cycles"
    Just xs -> concatMap (\x -> DAG.outgoingEdges x dag) xs


-- -- | Return the list of the IDs of the root nodes in the given DAG.
-- rootNodes :: DAG a -> [DAG.NodeID]
-- rootNodes dag = do
--   nodeID <- DAG.dagNodes dag
--   guard . empty $ DAG.ingoingEdges dag
--   return nodeID
--
--
-- -- | Return the list of the IDs of the leaf nodes in the given DAG.
-- leafNodes :: DAG a -> [DAG.NodeID]
-- leafNodes dag = do
--   nodeID <- DAG.dagNodes dag
--   guard . empty $ DAG.outgoingEdges dag
--   return nodeID