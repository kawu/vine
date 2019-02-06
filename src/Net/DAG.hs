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

import           Data.Ord (comparing)
import qualified Data.List as L
import qualified Data.Map.Strict as M
import qualified Data.DAG as DAG

import qualified Prelude.Backprop as PB
import qualified Numeric.Backprop as BP
import           Numeric.Backprop ((^^.))
import qualified Numeric.LinearAlgebra.Static.Backprop as LBP
import           Numeric.LinearAlgebra.Static.Backprop
  (R, L, BVar, Reifies, W, (#), (#>))
import qualified Numeric.LinearAlgebra.Static as LA

import           Net.Basic
import qualified Net.List as NL
import qualified Net.FeedForward as FFN
import           Net.FeedForward (FFN(..))
import qualified GradientDescent as GD

import Debug.Trace (trace)


----------------------------------------------
-- Recurrent DAG Network
----------------------------------------------


-- | Encoder RNN
--
-- NOTE: There is some bug in the underlying libraries; because of it, we need
-- to use `_matP :: L 2 5` instead `_matP :: L 1 5`, for instance.
--
data RecDAG = RecDAG
--   { _matA :: L
--       5   -- Size of the hidden state
--       10  -- Size of the past state + size of the input word embeddings
--     -- ^ To calculate the hidden state of the current arc given the past and
--     -- the vector representation of the current arc
  { _ffA :: FFN
      10  -- Size of the past state + size of the input word embeddings
      5   -- FFN's internal hidden state
      5   -- Size of the hidden state
    -- ^ To calculate the hidden state of the current arc given the past and
    -- the vector representation of the current arc
  , _matP :: L
      2   -- Output: single value
      5   -- Size of the hidden state
    -- ^ To calculate the probability of the given arc based on the its hidden
    -- representation and its vector representation
--   , _ffB :: FFN
--       10  -- Size of the hidden state + size of the input word embeddings
--       5   -- FFN's internal hidden state
--       1   -- Output: single value
--     -- ^ To calculate the probability of the given arc based on the its hidden
--     -- representation and its vector representation
  , _matI :: L
      2   -- Output: single value (relative importance)
      10  -- Size of the hidden state + size of the input word embeddings
    -- ^ To calculate the relative importance of the previous arc w.r.t.
    -- the current arc
--   , _ffI :: FFN
--       10  -- Size of the hidden state + size of the input word embeddings
--       5   -- FFN's internal hidden state
--       1   -- Output: single value (relative importance)
--     -- ^ To calculate the relative importance of the previous arc w.r.t.
--     -- the current arc
  , _p0  :: R 5
    -- ^ The initial past state
  } deriving (Show, Generic)

instance BP.Backprop RecDAG
makeLenses ''RecDAG


-- | Create a new, random network
new :: IO RecDAG
new = RecDAG
  -- <$> matrix 5 10
  <$> FFN.new 10 5 5
  <*> matrix 2 5  -- FFN.new 10 5 1
  <*> matrix 2 10 -- FFN.new 10 5 1
  <*> vector 5


-- | DAG with labeled arcs
type DAG a = DAG.DAG () a


-- | Run the network over a DAG labeled with input word embeddings.
run
  :: (Reifies s W)
  => BVar s RecDAG
    -- ^ Recurrent DAG network
  -> DAG (BVar s (R 5))
    -- ^ Input DAG with word embeddings
  -> M.Map DAG.EdgeID (BVar s (R 5), BVar s Double)
    -- ^ Output map with (i) hidden states and (ii) ,,probabilities''
run net dag =
  go M.empty (topoEdges dag)
  where
    go hiddenMap [] = hiddenMap
    go hiddenMap (arcID : rest) =
      let arcHidden = calculateHiddenState hiddenMap arcID
       in go (M.insert arcID arcHidden hiddenMap) rest
    calculateHiddenState hiddenMap arcID =
      let
        -- retrieve the input word embedding assigned to the current arc
        arc = DAG.edgeLabel arcID dag
        -- retrieve the hidden representations of the preceding arcs
        hids = map (fst . (hiddenMap M.!)) (DAG.prevEdges arcID dag)
        -- calculate the relative importance of the individual preceding arcs
        relImps = map
          (elem0 .
            -- (\hid -> FFN.run (net ^^. ffI) (hid # arc))
            (\hid -> (net ^^. matI) #> (hid # arc))
          )
          hids
        -- calculate ,,attention'' of each preceding arc (TODO: perhaps could
        -- be done more efficiently without `sequenceVar`?)
        atts = BP.sequenceVar . NL.softmax $ BP.collectVar relImps
        -- calculate the overall representation of the past
        past =
          if null hids
             then net ^^. p0
             else PB.sum . BP.collectVar $ do
                    (hid, att) <- zip hids atts
                    return $ LBP.vmap (*att) hid
        -- calculate the new hidden state
        -- newHid = (net ^^. matA) #> (past # arc)
        newHid = FFN.run (net ^^. ffA) (past # arc)
        -- newHid = logistic $ (net ^^. matA) #> (past # arc)
        -- calculate the ,,marginal probability''
        prob = logistic . elem0 $ (net ^^. matP) #> newHid
        -- prob = elem0 $ (net ^^. matP) #> newHid
      in
        (newHid, prob)


-- | Evaluate the network over a DAG labeled with input word embeddings.
-- User-friendly (and without back-propagation) version of `run`.
eval
  :: RecDAG
    -- ^ Recurrent DAG network
  -> DAG (R 5)
    -- ^ Input DAG with word embeddings
  -> M.Map DAG.EdgeID Double
eval net dag =
  BP.evalBP0
    ( BP.collectVar
    . fmap snd
    $ run
        (BP.constVar net)
        (fmap BP.constVar dag)
    )


----------------------------------------------
-- Training dataset
----------------------------------------------


-- | Training dataset: pairs of (i) DAGs and (ii) the target values
type Train =
  [ ( DAG (R 5)
    , M.Map DAG.EdgeID Double )
  ]


-- vocabulary, including special SOS/EOS symbol
noun, verb, adj, adv :: R 5
noun = LA.vector [1, 0, 0, 0, 0]
verb = LA.vector [0, 1, 0, 0, 0]
adj  = LA.vector [0, 0, 1, 0, 0]
adv  = LA.vector [0, 0, 0, 1, 0]
os   = LA.vector [0, 0, 0, 0, 1]


-- | TODO: Make sure this is correct (cf. `fromEdgesUnsafe`)
trainData :: Train
trainData =
  [ mkElem
      [ (edge 0 1 noun, 1)
      , (edge 1 2 verb, 1)
      , (edge 2 3 noun, 1)
      ]
  , mkElem
      [ (edge 0 1 noun, 1)
      , (edge 1 2 verb, 1)
      , (edge 2 3 adj, 1)
      , (edge 2 4 verb, 0)
      , (edge 3 4 noun, 1)
      , (edge 1 4 adj, 0)
      ]
  , mkElem
      [ (edge 0 1 adv, 0)
      , (edge 0 2 noun, 1)
      , (edge 2 3 verb, 1)
      ]
  , mkElem
      [ (edge 0 1 adj, 1)
      , (edge 0 5 adv, 0)
      , (edge 1 2 noun, 1)
      , (edge 1 3 adj, 0)
      , (edge 2 3 verb, 1)
      , (edge 3 4 adj, 1)
      , (edge 4 5 noun, 1)
      ]
  ]
  where
    edge p q x = DAG.Edge (DAG.NodeID p) (DAG.NodeID q) x
    mkElem desc0 =
      let desc = L.sortBy (comparing $ DAG.tailNode . fst) desc0
       in (mkDAG desc, mkTarget desc)
    mkDAG = DAG.fromEdgesUnsafe . map fst
    mkTarget = M.fromList . zip (map DAG.EdgeID [0..]) . map snd


----------------------------------------------
-- Error
----------------------------------------------


-- | Squared error between the target and the actual output.
errorOne
  :: (Reifies s W)
  => M.Map DAG.EdgeID (BVar s Double)
    -- ^ Target values
  -> M.Map DAG.EdgeID (BVar s Double)
    -- ^ Output values
  -> BVar s Double
errorOne target output = PB.sum . BP.collectVar $ do
  (arcID, tval) <- M.toList target
  let oval = output M.! arcID
  return $ (tval - oval) ^ 2
  -- return . abs $ tval - oval


-- | Error on a dataset.
errorMany
  :: (Reifies s W)
  => [M.Map DAG.EdgeID (BVar s Double)] -- ^ Targets
  -> [M.Map DAG.EdgeID (BVar s Double)] -- ^ Outputs
  -> BVar s Double
errorMany targets outputs =
  go targets outputs
  where
    go ts os =
      case (ts, os) of
        (t:tr, o:or) -> errorOne t o + go tr or
        ([], []) -> 0
        _ -> error "errorMany: lists of different size"


-- | Network error on a given dataset.
netError
  :: (Reifies s W)
  => Train
  -> BVar s RecDAG
  -> BVar s Double
netError dataSet net =
  let
    inputs = map fst dataSet
    outputs = map (fmap snd . run net . fmap BP.auto) inputs
    targets = map (fmap BP.auto . snd) dataSet
  in  
    errorMany targets outputs


----------------------------------------------
-- Training
----------------------------------------------


-- | Train with a custom dataset.
trainWith dataSet net =
  GD.gradDesc net (gdCfg dataSet)


-- | Train with the default dataset.
train net =
  trainWith trainData net


-----------------------------------------------------
-- Gradient descent
-----------------------------------------------------


-- | Gradient descent configuration
gdCfg dataSet = GD.Config
  { iterNum = 5000
  , scaleCoef = 0.05
  , gradient = BP.gradBP (netError dataSet)
  , substract = subRecDAG
  , quality = BP.evalBP (netError dataSet)
  , reportEvery = 500
  }


-- | Substract the second network (multiplied by the given coefficient) from
-- the first one.
subRecDAG x y coef = RecDAG
  -- { _matA = _matA x - scaleL (_matA y)
  { _ffA = FFN.substract (_ffA x) (_ffA y) coef
  , _matP = _matP x - scaleL (_matP y)
  , _matI = _matI x - scaleL (_matI y)
  , _p0   =   _p0 x - scaleR   (_p0 y)
  }
  where
    scaleL = LA.dmmap (*coef)
    scaleR = LA.dvmap (*coef)


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
